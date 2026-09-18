package com.ailearn.alibaba.eval;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * <h1>RetrievalMetrics — 检索层评估指标</h1>
 *
 * <p>评估 RAG 必须<b>分层</b>：检索质量与生成质量是两个独立的问题。</p>
 *
 * <h2>为什么必须分开评？</h2>
 * <p>如果只评最终答案，一旦效果不好，你无法知道是：</p>
 * <ul>
 *   <li>检索没找到正确文档（检索问题）→ 需要调切分、召回、重排</li>
 *   <li>检索找对了但模型没用（生成问题）→ 需要调 Prompt、换模型</li>
 * </ul>
 * <p>这两类问题的解法完全不同。混在一起评估等于放弃了归因能力。</p>
 *
 * <h2>本项目实现的指标</h2>
 *
 * <h3>1. Recall@K（召回率）</h3>
 * <pre>
 *   Recall@K = 前 K 个结果中命中的相关文档数 / 全部相关文档数
 * </pre>
 * <p><b>RAG 中最重要的指标</b>。因为检索是生成的输入——
 * 如果正确内容压根没被召回，后面再强的模型也无能为力。
 * 这是<b>召回上限</b>，一旦丢失无法在下游弥补。</p>
 *
 * <h3>2. Precision@K（精确率）</h3>
 * <pre>
 *   Precision@K = 前 K 个结果中命中的相关文档数 / K
 * </pre>
 * <p>衡量召回结果的「纯度」。精确率低意味着大量无关内容
 * 被塞进上下文，会稀释模型注意力并推高 token 成本。</p>
 *
 * <h3>3. MRR（平均倒数排名）</h3>
 * <pre>
 *   MRR = (1/|Q|) · Σ (1 / rank_i)
 *
 *   rank_i = 第 i 个问题中，第一个相关结果所在的排名
 * </pre>
 * <p>只关心「第一个正确结果排多前」。适合「只有一个正确答案」
 * 的场景（如事实性问答）。范围 (0, 1]，越大越好。</p>
 *
 * <h3>4. NDCG@K（归一化折损累计增益）</h3>
 * <pre>
 *   DCG@K  = Σ (2^rel_i - 1) / log2(i + 1)
 *   IDCG@K = 理想排序下的 DCG
 *   NDCG@K = DCG@K / IDCG@K
 * </pre>
 * <p>最严谨的排序质量指标。与 MRR 的区别在于它<b>考虑所有相关结果的位置</b>，
 * 而不仅是第一个。且通过对数折损体现「排在后面的收益递减」。</p>
 *
 * @author ai-learn
 */
public class RetrievalMetrics {

    /**
     * 计算 Recall@K。
     *
     * @param retrieved      检索结果列表，按相关性降序
     * @param k              截断位置
     * @param totalRelevant  全部相关文档的总数（评估样本中标注的相关分片数）
     * @return Recall@K，范围 [0, 1]
     */
    public static double recallAtK(List<RetrievedItem> retrieved, int k, int totalRelevant) {
        if (totalRelevant <= 0) {
            return 1.0;  // 无相关文档，视为无需召回
        }
        int hits = 0;
        int limit = Math.min(k, retrieved.size());
        for (int i = 0; i < limit; i++) {
            if (retrieved.get(i).relevant()) {
                hits++;
            }
        }
        return (double) hits / totalRelevant;
    }

    /**
     * 计算 Precision@K。
     */
    public static double precisionAtK(List<RetrievedItem> retrieved, int k) {
        if (k <= 0) {
            return 0.0;
        }
        int hits = 0;
        int limit = Math.min(k, retrieved.size());
        for (int i = 0; i < limit; i++) {
            if (retrieved.get(i).relevant()) {
                hits++;
            }
        }
        return (double) hits / k;
    }

    /**
     * 计算单个查询的倒数排名（Reciprocal Rank）。
     *
     * @return 第一个相关结果的排名的倒数；无相关结果时返回 0
     */
    public static double reciprocalRank(List<RetrievedItem> retrieved) {
        for (int i = 0; i < retrieved.size(); i++) {
            if (retrieved.get(i).relevant()) {
                return 1.0 / (i + 1);
            }
        }
        return 0.0;
    }

    /**
     * 计算单个查询的 NDCG@K。
     *
     * @param retrieved     检索结果（已按模型给出的顺序排列）
     * @param k             截断位置
     * @param totalRelevant 相关文档总数，用于构造理想排序
     */
    public static double ndcgAtK(List<RetrievedItem> retrieved, int k, int totalRelevant) {
        if (totalRelevant <= 0 || retrieved.isEmpty() || k <= 0) {
            return 1.0;
        }

        // ---- 实际 DCG ----
        // 本项目采用二元相关性（相关=1 / 不相关=0），
        // 因此 gain = 2^1 - 1 = 1。
        double dcg = 0;
        int limit = Math.min(k, retrieved.size());
        for (int i = 0; i < limit; i++) {
            if (retrieved.get(i).relevant()) {
                // 位置 i（从 0 开始）对应排名 i+1
                dcg += 1.0 / (Math.log(i + 2) / Math.log(2));
            }
        }

        // ---- 理想 DCG ----
        // 理想情况：所有相关结果都排在最前面
        double idcg = 0;
        int idealHits = Math.min(totalRelevant, k);
        for (int i = 0; i < idealHits; i++) {
            idcg += 1.0 / (Math.log(i + 2) / Math.log(2));
        }

        return idcg == 0 ? 0.0 : dcg / idcg;
    }

    /**
     * 汇总多个查询的指标，返回平均值。
     *
     * <p>单查询指标波动很大，只有平均值才有决策意义。
     * 建议评估集至少 30 个样本，50~100 个更可靠。</p>
     */
    public static double mean(List<Double> values) {
        if (values == null || values.isEmpty()) {
            return 0.0;
        }
        return values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }

    /**
     * 检索结果项：携带相关性判定。
     *
     * @param id       文档标识
     * @param relevant 是否相关（由金标准判定）
     * @param score    检索器给出的分数，用于调试
     * @param preview  文本预览，便于人工核对误判
     */
    public record RetrievedItem(String id, boolean relevant, double score, String preview) {}

    /**
     * 根据疑似重复的文本计算「去重后」的相关文档数。
     *
     * <p>分片切分可能把同一段内容切成多块，导致 totalRelevant 被高估，
     * 进而虚低 Recall。这里提供一个去重工具。</p>
     */
    public static int distinctCount(List<String> texts) {
        if (texts == null) {
            return 0;
        }
        Set<String> normalized = new HashSet<>();
        for (String t : texts) {
            if (t != null) {
                normalized.add(t.trim().toLowerCase());
            }
        }
        return normalized.size();
    }

    /** 把若干个单查询指标汇总为报告行 */
    public static String format(double recall, double precision, double mrr, double ndcg) {
        List<String> parts = new ArrayList<>();
        parts.add(String.format("Recall=%.3f", recall));
        parts.add(String.format("Precision=%.3f", precision));
        parts.add(String.format("MRR=%.3f", mrr));
        parts.add(String.format("NDCG=%.3f", ndcg));
        return String.join("  ", parts);
    }
}
