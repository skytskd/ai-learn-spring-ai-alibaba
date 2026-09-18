package com.ailearn.alibaba.rag;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * <h1>QueryRewriter — 查询改写与扩展</h1>
 *
 * <p>RAG 调优中最容易被忽视、但收益极高的一环：<b>在检索之前先优化查询本身</b>。</p>
 *
 * <h2>为什么需要查询改写？</h2>
 * <p>用户的原始提问往往是「口语化、省略、指代不明」的。例如：</p>
 * <table border="1">
 *   <tr><th>用户原话</th><th>检索失败的原因</th></tr>
 *   <tr><td>"它怎么收费？"</td><td>「它」指代不明，向量编码后语义空洞</td></tr>
 *   <tr><td>"那个embedding模型"</td><td>缺少关键实体，无法锁定具体文档</td></tr>
 *   <tr><td>"太贵了怎么办"</td><td>口语表达与文档书面语之间词汇鸿沟</td></tr>
 * </table>
 *
 * <h2>本类实现的三种策略</h2>
 *
 * <h3>1. HyDE（Hypothetical Document Embeddings）</h3>
 * <p>论文：Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"
 * (arXiv 2212.10496)</p>
 *
 * <p>核心洞察：<b>问题与答案在向量空间中往往距离较远</b>。
 * 用户问「怎么收费」，文档写的是「按量付费，qwen-turbo 约 ¥0.3/千Token」——
 * 两者字面不同、语义分布也不同。</p>
 *
 * <p>HyDE 的做法是：先让 LLM <b>编造一段假想的答案文档</b>，
 * 再用这段假答案去检索。因为假答案在「文体、词汇、句式」上都接近真实文档，
 * 向量距离显著更近。即使假答案内容有事实错误也无妨——我们要的是它的
 * <b>向量表示</b>，不是它的内容。</p>
 *
 * <pre>
 *   原始问题 ──LLM──→ 假想答案文档 ──Embedding──→ 检索
 *   "怎么收费"        "百炼采用按量付费模式，     向量更接近
 *                      qwen-turbo 约 0.3 元..."    真实文档
 * </pre>
 *
 * <h3>2. 多查询扩展（Multi-Query Expansion）</h3>
 * <p>从不同角度生成 N 个改写版本，分别检索后合并。
 * 单个查询只能覆盖一个「语义视角」，多查询能显著提升召回率——
 * 这正是「召回求全、精排求准」原则中「求全」的手段。</p>
 *
 * <h3>3. 指代消解（Coreference Resolution）</h3>
 * <p>把「它」「这个」「那个」替换成明确实体，避免检索时语义丢失。</p>
 *
 * @author ai-learn
 */
@Slf4j
@Component
public class QueryRewriter {

    private final ChatClient chatClient;

    public QueryRewriter(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    /**
     * HyDE：生成假想答案文档。
     *
     * <p>返回的文本<b>不用于展示给用户</b>，只用于生成检索向量。</p>
     *
     * @param query 用户原始问题
     * @return 假想答案文档；生成失败时返回原查询（降级为不使用 HyDE）
     */
    public String hypothesize(String query) {
        String prompt = """
                请针对下面的问题，写一段 150 字左右的「假设性答案」。

                要求：
                1. 假设你是在编写一份技术文档，用陈述句、书面语描述
                2. 直接给出答案内容，不要出现「我认为」「可能」等不确定表述
                3. 不要解释、不要道歉、不要说明这是假设
                4. 内容允许不完全准确，重点是用词和文体要像技术文档

                问题：%s
                """.formatted(query);

        try {
            String result = chatClient.prompt()
                    .user(prompt)
                    .call()
                    .content();

            if (result == null || result.isBlank()) {
                return query;
            }
            log.debug("[HyDE] 生成假想文档 {} 字", result.length());
            return result;

        } catch (Exception e) {
            log.warn("[HyDE] 生成失败，退化使用原查询: {}", e.getMessage());
            return query;
        }
    }

    /**
     * 多查询扩展：生成语义等价但表述不同的查询变体。
     *
     * <p>返回结果<b>包含原查询</b>，保证原有召回能力不丢失。</p>
     *
     * @param query 用户原始问题
     * @param count 期望生成的变体数量（不含原查询）
     * @return 查询列表，第 0 个元素恒为原查询
     */
    public List<String> expand(String query, int count) {
        List<String> queries = new ArrayList<>();
        queries.add(query);

        if (count <= 0) {
            return queries;
        }

        String prompt = """
                请将下面的问题改写成 %d 个不同的检索查询。

                要求：
                1. 每个改写都从不同角度切入，覆盖问题可能的不同意图
                2. 使用更正式、更接近技术文档的书面语
                3. 补充问题中可能省略的关键词
                4. 每行一个，不要编号，不要任何解释

                原问题：%s
                """.formatted(count, query);

        try {
            String result = chatClient.prompt()
                    .user(prompt)
                    .call()
                    .content();

            if (result != null && !result.isBlank()) {
                result.lines()
                        .map(String::trim)
                        // 过滤空行、列表符号、编号前缀
                        .filter(line -> !line.isEmpty())
                        .map(line -> line.replaceFirst("^[-*•]\\s*", ""))
                        .map(line -> line.replaceFirst("^\\d+[.、)]\\s*", ""))
                        .filter(line -> line.length() > 1)
                        .limit(count)
                        .forEach(queries::add);
            }

            log.debug("[QueryExpand] {} → {} 个查询", query, queries.size());

        } catch (Exception e) {
            log.warn("[QueryExpand] 扩展失败，仅使用原查询: {}", e.getMessage());
        }

        return queries;
    }
}
