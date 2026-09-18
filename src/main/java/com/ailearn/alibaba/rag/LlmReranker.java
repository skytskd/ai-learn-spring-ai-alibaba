package com.ailearn.alibaba.rag;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * <h1>LlmReranker — LLM 重排序（Cross-Encoder 思路）</h1>
 *
 * <p>检索流水线的最后一道精排关卡，也是<b>投入产出比最高的一步</b>。</p>
 *
 * <h2>为什么需要重排序？</h2>
 * <p>向量检索属于 <b>Bi-Encoder（双塔）</b>架构：query 和 document
 * 分别独立编码成向量，再算余弦相似度。因为两者从未"见过面"，
 * 无法建模细粒度的交互关系，只能捕捉粗粒度语义。</p>
 *
 * <p>重排序属于 <b>Cross-Encoder（交叉编码器）</b>：把 query 和
 * document <b>拼在一起</b>送入模型，让注意力机制在两者之间充分交互。
 * 精度显著更高，但计算成本也高得多（无法预计算，每个候选都要跑一次前向）。</p>
 *
 * <pre>
 *   Bi-Encoder（快而粗）              Cross-Encoder（慢而准）
 *   ┌───────┐   ┌───────┐            ┌──────────────────────┐
 *   │ query │   │  doc  │            │ query [SEP] doc      │
 *   └───┬───┘   └───┬───┘            └──────────┬───────────┘
 *       ↓           ↓                           ↓
 *    Encoder     Encoder                    Encoder
 *       ↓           ↓                           ↓
 *    vec_q       vec_d                      相关分数
 *       └────┬──────┘                     （已建模交互）
 *       余弦相似度
 * </pre>
 *
 * <h2>典型两阶段架构</h2>
 * <pre>
 *   全量文档 ──混合检索──→ Top-20 ──重排序──→ Top-5 ──→ 送入 LLM 生成
 *             （召回，求全）           （精排，求准）
 * </pre>
 *
 * <h2>本实现的取舍</h2>
 * <p>生产环境应使用专用重排模型（BGE-Reranker-v2 / Cohere Rerank /
 * 阿里云 gte-rerank），延迟低、成本低、效果好。本类用 LLM 打分是为了
 * <b>零额外依赖、便于教学演示</b>，且能直观看到「重排前后的排名变化」。</p>
 *
 * <p>⚠️ 性能警告：LLM 重排会为每个候选发送一次请求。候选数应控制在
 * 20 以内，否则延迟和成本急剧上升。</p>
 *
 * @author ai-learn
 */
@Slf4j
@Component
public class LlmReranker {

    private final ChatClient chatClient;

    public LlmReranker(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    /**
     * 对候选文档做相关性精排。
     *
     * @param query      用户查询
     * @param candidates 候选文档（通常来自 RRF 融合结果）
     * @param topN       精排后保留的条数
     * @return 按 LLM 判定相关性降序排列的结果
     */
    public List<ScoredDocument> rerank(String query, List<ScoredDocument> candidates, int topN) {
        if (candidates == null || candidates.isEmpty()) {
            return List.of();
        }

        // 候选已足够少时无需精排，直接截断——避免无谓的模型调用
        if (candidates.size() <= topN) {
            log.debug("[Rerank] 候选数 {} ≤ topN {}，跳过重排序", candidates.size(), topN);
            return candidates.stream()
                    .map(sd -> sd.withSource(ScoredDocument.Source.RERANK))
                    .toList();
        }

        log.debug("[Rerank] 开始精排：{} 个候选 → Top-{}", candidates.size(), topN);

        List<ScoredDocument> scored = candidates.stream()
                .map(candidate -> {
                    double score = scoreOne(query, candidate.text());
                    return ScoredDocument.of(candidate.document(), score, ScoredDocument.Source.RERANK);
                })
                // 稳定排序：分数相同时保持原有相对顺序
                .sorted((a, b) -> Double.compare(b.score(), a.score()))
                .limit(topN)
                .toList();

        if (log.isDebugEnabled()) {
            log.debug("[Rerank] 排名变化：");
            for (int i = 0; i < scored.size(); i++) {
                ScoredDocument sd = scored.get(i);
                int oldRank = indexOf(candidates, sd.id()) + 1;
                log.debug("  #{} (原 #{}) 分数 {} — {}",
                        i + 1, oldRank, String.format("%.4f", sd.score()), truncate(sd.text(), 40));
            }
        }

        return scored;
    }

    /**
     * 用 LLM 给单个「查询-文档」对打相关性分数。
     *
     * <h3>Prompt 设计要点</h3>
     * <ol>
     *   <li><b>强制纯数字输出</b>——避免模型返回「相关性很高」这类文字，
     *       否则解析失败。明确要求"只输出一个数字"。</li>
     *   <li><b>temperature = 0</b>——打分任务必须确定性，否则同一输入
     *       两次调用结果不同，评估会不稳定。</li>
     *   <li><b>给出评分标准</b>——把 0~10 各档含义写清楚，
     *       比只说"打分"的稳定性高得多。</li>
     * </ol>
     *
     * @return 归一化到 [0,1] 的相关性分数；解析失败时返回 0，使该文档沉底
     */
    private double scoreOne(String query, String docText) {
        String prompt = """
                你是一个检索相关性评估专家。请评估下面【文档】与【查询】的相关程度。

                【查询】
                %s

                【文档】
                %s

                【评分标准】
                - 10 分：文档直接、完整地回答了查询的核心问题
                - 7-9 分：文档包含查询所需的关键信息，但不够完整
                - 4-6 分：文档主题相关，部分信息可用
                - 1-3 分：仅有个别词语相关，实质内容无关
                - 0 分：完全不相关

                只输出一个 0 到 10 之间的整数，不要输出任何其他文字、解释或标点。
                """.formatted(truncate(query, 500), truncate(docText, 1500));

        try {
            String raw = chatClient.prompt()
                    .user(prompt)
                    .call()
                    .content();

            int value = parseScore(raw);
            return value / 10.0;

        } catch (Exception e) {
            // 重排失败不应中断整个 RAG 链路——降级为 0 分，让该文档沉底
            log.warn("[Rerank] 打分失败，该文档降级处理: {}", e.getMessage());
            return 0.0;
        }
    }

    /**
     * 从 LLM 原始输出中稳健地解析分数。
     *
     * <p>即使 Prompt 要求只输出数字，模型仍可能返回 "8分"、"评分：8"、
     * "```\n8\n```" 等变体。因此用正则扫描第一个出现的 0-10 数字。</p>
     */
    private int parseScore(String raw) {
        if (raw == null || raw.isBlank()) {
            return 0;
        }
        java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("(10|[0-9])")
                .matcher(raw.trim());
        if (m.find()) {
            return Integer.parseInt(m.group(1));
        }
        log.warn("[Rerank] 无法解析分数，原始输出: {}", truncate(raw, 60));
        return 0;
    }

    private int indexOf(List<ScoredDocument> list, String id) {
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i).id().equals(id)) {
                return i;
            }
        }
        return list.size();
    }

    private String truncate(String s, int max) {
        if (s == null) {
            return "";
        }
        return s.length() <= max ? s : s.substring(0, max) + "...";
    }
}
