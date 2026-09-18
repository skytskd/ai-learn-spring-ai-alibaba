package com.ailearn.alibaba.memory;

import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * <h1>ConversationState — 会话状态（分层记忆的载体）</h1>
 *
 * <p>一个会话的完整记忆快照。它是「多轮记忆」中最核心的数据结构。</p>
 *
 * <h2>三层结构</h2>
 * <pre>
 *   ┌─────────────────────────────────────────────────────┐
 *   │  Layer 3  长期事实（longTermFacts）                  │
 *   │  跨会话稳定不变的信息：用户姓名、职业、技术栈偏好       │
 *   │  例："用户叫张三"、"用户是 Java 开发工程师"            │
 *   ├─────────────────────────────────────────────────────┤
 *   │  Layer 2  会话摘要（summary）                        │
 *   │  已被压缩的历史对话要点，控制 token 占用               │
 *   │  例："用户询问了 Spring AI 的 RAG 实现方式..."        │
 *   ├─────────────────────────────────────────────────────┤
 *   │  Layer 1  近期消息窗口（recentMessages）              │
 *   │  最近 N 轮原始对话，保留完整细节                       │
 *   │  例："用户: 我叫张三" / "助手: 你好张三！"             │
 *   └─────────────────────────────────────────────────────┘
 * </pre>
 *
 * <h2>为什么要分层？</h2>
 * <p>把全部历史对话原样塞进上下文，是初学者最常见的做法，也是
 * 最贵的做法。以 20 轮对话为例，每轮约 200 token，就是 4000 token——
 * 每问一次都要重发一遍，成本随轮次<b>线性增长</b>，且无关历史会
 * 稀释模型注意力，导致"越聊越傻"。</p>
 *
 * <p>分层的本质是<b>信息分级存储</b>：</p>
 * <ul>
 *   <li>近期细节最影响当前回答质量 → 保留原文（Layer 1）</li>
 *   <li>中期要点仍可能被引用 → 压缩成摘要（Layer 2）</li>
 *   <li>稳定属性跨会话有效 → 抽成结构化事实（Layer 3）</li>
 * </ul>
 *
 * @author ai-learn
 */
@Data
@NoArgsConstructor
public class ConversationState {

    /** 会话唯一标识 */
    private String conversationId;

    /** Layer 1：近期消息窗口，保留原文 */
    private List<MemoryMessage> recentMessages = new ArrayList<>();

    /** Layer 2：历史对话的滚动摘要 */
    private String summary = "";

    /** Layer 3：抽取出的长期事实（去重后的陈述句列表） */
    private List<String> longTermFacts = new ArrayList<>();

    /** 累计处理的用户消息轮次，用于触发摘要的节流控制 */
    private int totalTurns = 0;

    /** 摘要最近一次更新的轮次，用于避免每轮都调用 LLM 做摘要 */
    private int lastSummarizedAtTurn = 0;

    /** 会话创建时间 */
    private Instant createdAt = Instant.now();

    /** 最近一次访问时间，用于空闲会话清理 */
    private Instant lastAccessedAt = Instant.now();

    public ConversationState(String conversationId) {
        this.conversationId = conversationId;
    }

    /**
     * 追加一条消息到近期窗口。
     */
    public void addMessage(MemoryMessage message) {
        if (recentMessages == null) {
            recentMessages = new ArrayList<>();
        }
        recentMessages.add(message);
        this.lastAccessedAt = Instant.now();
    }

    /**
     * 从窗口头部移除最旧的消息，直到窗口大小不超过 maxSize。
     *
     * <p>注意这里移除的是「最旧」的消息——这是滑动窗口（FIFO）
     * 的基本语义。被移出的消息并非丢失，而是应该已经
     * 被摘要吸收进 Layer 2。</p>
     *
     * @param maxSize 窗口保留的最大消息条数
     * @return 被移出的消息（调用方可在摘要前用它们作为素材）
     */
    public List<MemoryMessage> evictOldest(int maxSize) {
        List<MemoryMessage> evicted = new ArrayList<>();
        if (recentMessages == null) {
            return evicted;
        }

        while (recentMessages.size() > maxSize) {
            // 移除索引 0（最旧），为保持语义清晰不用 removeFirst()
            evicted.add(recentMessages.remove(0));
        }
        return evicted;
    }

    /**
     * 添加一条长期事实，自动去重。
     *
     * <p>去重是必要的：LLM 抽取事实时容易在不同轮次重复提取同一信息，
     * 若不控制，事实列表会迅速膨胀，最终把上下文挤爆。</p>
     *
     * @return true 表示确实是新事实，false 表示已存在被跳过
     */
    public boolean addFact(String fact) {
        if (fact == null || fact.isBlank()) {
            return false;
        }
        String normalized = fact.trim();
        if (longTermFacts == null) {
            longTermFacts = new ArrayList<>();
        }

        // 简单去重：忽略大小写与首尾标点差异
        for (String existing : longTermFacts) {
            if (existing.equalsIgnoreCase(normalized)) {
                return false;
            }
        }

        longTermFacts.add(normalized);
        return true;
    }
}
