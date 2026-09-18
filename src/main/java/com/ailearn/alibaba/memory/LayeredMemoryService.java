package com.ailearn.alibaba.memory;

import com.ailearn.alibaba.config.MemoryProperties;
import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * <h1>LayeredMemoryService — 分层记忆服务</h1>
 *
 * <p>实现完整的「滑动窗口 + 摘要压缩 + 长期事实」三层记忆架构。</p>
 *
 * <h2>工作流程</h2>
 * <pre>
 *   用户: "我叫张三，是Java工程师"
 *      │
 *      ▼
 *   [1] 追加到 Layer 1 近期窗口
 *      │
 *      ▼
 *   [2] 检查窗口是否超限？
 *      │    ├─ 否 → 直接组装上下文，返回
 *      │    └─ 是 → 触发摘要
 *      │             ├─ 把溢出的旧消息喂给 LLM → 滚动更新 Layer 2 摘要
 *      │             └─ 同时抽取 Layer 3 长期事实（节流执行）
 *      ▼
 *   [3] 组装三段式上下文：
 *       [长期事实] + [历史摘要] + [近期对话原文]
 *      │
 *      ▼
 *   [4] 调用 LLM，把助手回复也写入 Layer 1
 * </pre>
 *
 * <h2>滚动摘要（Rolling Summary）的关键设计</h2>
 * <p>摘要不是「重新总结全部历史」，而是<b>「旧摘要 + 新溢出内容 → 新摘要」</b>。
 * 这样每次摘要的输入规模是恒定的，不会随对话轮次增长，
 * 避免了摘要本身成为性能瓶颈。</p>
 *
 * <h2>摘要触发节流</h2>
 * <p>如果每轮都调用 LLM 做摘要，成本和延迟都不可接受。
 * 因此引入两个约束：</p>
 * <ol>
 *   <li><b>窗口占用触发</b>：只有消息数超过阈值才可能触发</li>
 *   <li><b>轮次间隔节流</b>：距上次摘要至少间隔 N 轮（{@code summaryInterval}）</li>
 * </ol>
 *
 * @author ai-learn
 */
@Slf4j
@Component
public class LayeredMemoryService {

    private final ChatClient chatClient;
    private final MemoryProperties properties;

    /**
     * 会话状态存储。
     *
     * <p>使用 {@link ConcurrentHashMap} 而非普通 HashMap：
     * Web 容器是多线程环境，同一用户可能并发发起请求，
     * 非线程安全容器会导致数据损坏或死循环。</p>
     *
     * <p>⚠️ 这是<b>进程内内存存储</b>，重启即丢失，且无法水平扩展
     * （多实例部署时会话不共享）。生产环境应替换为 Redis
     * （参考 {@link #describePersistenceOptions()}）。</p>
     */
    private final Map<String, ConversationState> store = new ConcurrentHashMap<>();

    /** 空闲会话清理线程 */
    private final ScheduledExecutorService cleaner = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "memory-cleaner");
        t.setDaemon(true);
        return t;
    });

    public LayeredMemoryService(ChatClient chatClient, MemoryProperties properties) {
        this.chatClient = chatClient;
        this.properties = properties;

        // 定时清理空闲会话，防止内存无限增长（内存泄漏的经典来源）
        cleaner.scheduleAtFixedRate(this::evictIdleConversations, 10, 10, TimeUnit.MINUTES);
        log.info("[记忆] 分层记忆服务已启动：窗口 {} 条，摘要触发 {} 条，摘要间隔 {} 轮",
                properties.getMaxRecentMessages(),
                properties.getSummarizeThreshold(),
                properties.getSummaryInterval());
    }

    // ==================== 核心对话方法 ====================

    /**
     * 带分层记忆的对话。
     *
     * @param conversationId 会话 ID，相同 ID 共享记忆
     * @param userMessage    用户输入
     * @return 包含回答与记忆状态的结构化结果
     */
    public MemoryChatResult chat(String conversationId, String userMessage) {
        ConversationState state = store.computeIfAbsent(
                conversationId, ConversationState::new);

        long start = System.currentTimeMillis();

        // ---------- [1] 用户消息入窗口 ----------
        state.addMessage(MemoryMessage.user(userMessage));
        state.setTotalTurns(state.getTotalTurns() + 1);

        // ---------- [2] 压缩：溢出 → 摘要 ----------
        boolean summarized = compact(state);

        // ---------- [3] 组装三段式上下文 ----------
        String systemContext = buildSystemContext(state);

        // ---------- [4] 调用 LLM ----------
        String reply = chatClient.prompt()
                .user(userMessage)
                .system(systemContext)
                .call()
                .content();

        // ---------- [5] 助手回复入窗口 ----------
        state.addMessage(MemoryMessage.assistant(reply));
        // 助手消息入窗后可能再次超限，但不在本轮触发摘要——
        // 留到下一次用户发言时统一处理，避免单轮内两次 LLM 调用。
        state.evictOldest(properties.getMaxRecentMessages() * 2);

        long elapsed = System.currentTimeMillis() - start;

        return new MemoryChatResult(
                conversationId,
                reply,
                state.getSummary(),
                new ArrayList<>(state.getLongTermFacts()),
                state.getRecentMessages().size(),
                summarized,
                elapsed);
    }

    /**
     * 上下文压缩：把超出窗口的旧消息滚动进摘要，并抽取长期事实。
     *
     * @return 本次是否执行了摘要生成
     */
    private boolean compact(ConversationState state) {
        int current = state.getRecentMessages().size();

        // 约束 1：未达阈值不触发，避免短对话白白消耗 LLM 调用
        if (current <= properties.getSummarizeThreshold()) {
            return false;
        }

        // 约束 2：轮次间隔节流，防止连续多轮反复摘要
        int turnsSinceLastSummary = state.getTotalTurns() - state.getLastSummarizedAtTurn();
        if (turnsSinceLastSummary < properties.getSummaryInterval()) {
            // 未到摘要间隔，但仍需裁剪窗口，防止无限增长。
            // 这里直接丢弃最旧消息——它们将在下个摘要周期被一并处理，
            // 虽有信息损失，但换取了稳定的内存与延迟表现。
            state.evictOldest(properties.getMaxRecentMessages());
            return false;
        }

        // ---------- 执行滚动摘要 ----------
        // 溢出部分 = 窗口保留量之外的所有旧消息
        int keep = properties.getMaxRecentMessages() / 2;
        List<MemoryMessage> all = state.getRecentMessages();
        int overflowCount = Math.max(0, all.size() - keep);

        if (overflowCount == 0) {
            return false;
        }

        List<MemoryMessage> overflow = new ArrayList<>(all.subList(0, overflowCount));
        state.evictOldest(keep);

        String newSummary = summarize(state.getSummary(), overflow);
        if (newSummary != null && !newSummary.isBlank()) {
            state.setSummary(newSummary);
        }

        // ---------- 抽取长期事实 ----------
        if (properties.isEnableFactExtraction()) {
            extractFacts(state, overflow);
        }

        state.setLastSummarizedAtTurn(state.getTotalTurns());

        log.info("[记忆] 会话 {} 已压缩：{} 条消息并入摘要，当前摘要 {} 字，长期事实 {} 条",
                state.getConversationId(), overflow.size(),
                state.getSummary().length(), state.getLongTermFacts().size());

        return true;
    }

    /**
     * 滚动摘要：把「旧摘要 + 新增溢出消息」压缩成新摘要。
     *
     * <p>这是本类的核心。关键在于<b>增量式</b>——输入规模恒定，
     * 不随对话轮次增长。</p>
     */
    private String summarize(String oldSummary, List<MemoryMessage> overflow) {
        String conversationText = overflow.stream()
                .map(MemoryMessage::toPromptLine)
                .reduce((a, b) -> a + "\n" + b)
                .orElse("");

        String prompt = """
                请把下面的对话内容压缩成一段简洁的摘要。

                【已有的历史摘要】
                %s

                【新增的对话内容】
                %s

                【压缩要求】
                1. 把新内容与已有摘要合并，输出一段完整的更新后摘要
                2. 保留：用户的身份信息、明确的需求、已确认的结论、待办事项
                3. 丢弃：寒暄、重复表述、与后续无关的细节
                4. 用第三人称陈述，控制在 200 字以内
                5. 只输出摘要正文，不要任何前缀、标题或解释
                """.formatted(
                oldSummary == null || oldSummary.isBlank() ? "（暂无）" : oldSummary,
                conversationText);

        try {
            String result = chatClient.prompt().user(prompt).call().content();
            return result == null ? oldSummary : result.trim();
        } catch (Exception e) {
            // 摘要失败时保留旧摘要，不阻断对话主流程
            log.warn("[记忆] 摘要生成失败，保留原摘要: {}", e.getMessage());
            return oldSummary;
        }
    }

    /**
     * 从对话中抽取长期事实（Layer 3）。
     *
     * <p>与摘要的区别：摘要是「叙事性的」，事实是「结构化、可复用的」。
     * 例如摘要里写「用户介绍了自己的背景」，而事实是「用户叫张三」、
     * 「用户是 Java 开发工程师」——后者可以直接注入到任何后续请求中。</p>
     */
    private void extractFacts(ConversationState state, List<MemoryMessage> overflow) {
        String conversationText = overflow.stream()
                .map(MemoryMessage::toPromptLine)
                .reduce((a, b) -> a + "\n" + b)
                .orElse("");

        String prompt = """
                请从下面的对话中抽取关于【用户】的长期稳定事实。

                【对话内容】
                %s

                【抽取规则】
                1. 只抽取长期有效的事实：姓名、职业、技术栈、偏好、习惯、目标
                2. 不要抽取一次性信息：当前的临时问题、具体某次提问的内容
                3. 每条事实用一句简短的陈述句，以"用户"开头
                4. 如果没有可抽取的长期事实，输出"无"
                5. 每行一条，不要编号，不要任何解释

                示例输出：
                用户叫张三
                用户是一名 Java 开发工程师
                用户偏好简洁的技术回答
                """.formatted(conversationText);

        try {
            String result = chatClient.prompt().user(prompt).call().content();
            if (result == null || result.isBlank() || result.trim().equals("无")) {
                return;
            }

            int added = 0;
            for (String line : result.lines().toList()) {
                String fact = line.trim()
                        .replaceFirst("^[-*•]\\s*", "")
                        .replaceFirst("^\\d+[.、)]\\s*", "");
                if (fact.length() > 2 && state.addFact(fact)) {
                    added++;
                }
            }

            if (added > 0) {
                log.debug("[记忆] 抽取到 {} 条新的事实", added);
            }

        } catch (Exception e) {
            log.warn("[记忆] 事实抽取失败: {}", e.getMessage());
        }
    }

    /**
     * 组装三段式系统上下文。
     *
     * <p>顺序很重要：<b>稳定的信息在前，易变的对话在后</b>。
     * 这与 KV Cache 的复用机制有关——前缀越稳定，
     * 缓存命中率越高，推理速度越快、成本越低。</p>
     */
    private String buildSystemContext(ConversationState state) {
        StringBuilder sb = new StringBuilder();
        sb.append("你是一个专业的 AI 编程学习助手，名叫\"通义学伴\"。\n");
        sb.append("请使用中文回复，回答简洁清晰。\n\n");

        // Layer 3：长期事实
        if (!state.getLongTermFacts().isEmpty()) {
            sb.append("【关于用户的长期信息】\n");
            state.getLongTermFacts().forEach(f -> sb.append("- ").append(f).append("\n"));
            sb.append("\n");
        }

        // Layer 2：历史摘要
        if (state.getSummary() != null && !state.getSummary().isBlank()) {
            sb.append("【之前对话的摘要】\n")
                    .append(state.getSummary()).append("\n\n");
        }

        // Layer 1：近期对话原文
        List<MemoryMessage> recent = state.getRecentMessages();
        if (!recent.isEmpty()) {
            sb.append("【最近对话记录】\n");
            // 不含最后一条（它就是本次的用户输入，会通过 .user() 单独传入）
            int limit = Math.max(0, recent.size() - 1);
            for (int i = 0; i < limit; i++) {
                sb.append(recent.get(i).toPromptLine()).append("\n");
            }
        }

        return sb.toString();
    }

    // ==================== 会话管理 ====================

    /** 获取会话状态（只读用途，如前端展示记忆内容） */
    public ConversationState getState(String conversationId) {
        return store.get(conversationId);
    }

    /** 清空指定会话的记忆 */
    public boolean clear(String conversationId) {
        return store.remove(conversationId) != null;
    }

    /** 当前活跃会话数 */
    public int activeConversations() {
        return store.size();
    }

    /**
     * 清理空闲会话。
     *
     * <p>没有这一步，进程内记忆会无限增长——这是内存泄漏的典型来源。
     * 生产环境如果改用 Redis，则应设置 TTL 而非自己写清理逻辑。</p>
     */
    private void evictIdleConversations() {
        long ttlMillis = properties.getIdleTtlMinutes() * 60_000L;
        long now = System.currentTimeMillis();

        int before = store.size();
        store.entrySet().removeIf(e ->
                now - e.getValue().getLastAccessedAt().toEpochMilli() > ttlMillis);

        int removed = before - store.size();
        if (removed > 0) {
            log.info("[记忆] 清理空闲会话 {} 个，剩余 {} 个", removed, store.size());
        }
    }

    @PreDestroy
    public void shutdown() {
        cleaner.shutdownNow();
    }

    /**
     * 说明生产环境的持久化方案（教学用）。
     */
    public String describePersistenceOptions() {
        return """
                生产环境记忆持久化的三种方案：

                1. Redis（推荐）
                   - 结构：Hash 存储会话状态，或 List 存储消息序列
                   - 优势：读写快、原生 TTL、天然支持多实例共享
                   - 关键：设置 TTL 避免冷会话堆积，序列化用 JSON 或 Protobuf

                2. 关系型数据库（PostgreSQL / MySQL）
                   - 结构：conversation 表 + message 表
                   - 优势：可审计、可分析、支持复杂查询
                   - 劣势：写入延迟高于 Redis，需要自行处理会话淘汰

                3. 向量数据库（做语义记忆检索）
                   - 结构：把每轮对话向量化存入，检索时按语义召回相关历史
                   - 优势：可从海量历史中精准召回"与当前问题相关"的片段
                   - 适用：超长周期（数月）的助手类产品
                   - 注意：这是"记忆检索"，与"记忆存储"是两个不同层次的问题

                实践建议：Redis 做热存储 + 数据库做冷归档，是最常见的组合。
                Spring AI 已提供 ChatMemoryRepository 接口的多种实现，
                可通过依赖注入直接替换，无需改动业务代码。
                """;
    }

    /**
     * 分层记忆对话的结果载体。
     *
     * @param conversationId 会话 ID
     * @param reply          AI 回复
     * @param summary        当前摘要（Layer 2 快照）
     * @param longTermFacts  当前长期事实（Layer 3 快照）
     * @param windowSize     Layer 1 窗口内消息数
     * @param summarized     本轮是否触发了摘要压缩
     * @param elapsedMs      耗时毫秒
     */
    public record MemoryChatResult(
            String conversationId,
            String reply,
            String summary,
            List<String> longTermFacts,
            int windowSize,
            boolean summarized,
            long elapsedMs
    ) {}
}
