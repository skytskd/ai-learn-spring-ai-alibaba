package com.ailearn.alibaba.eval;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * <h1>EvalReport — 评估报告</h1>
 *
 * <p>评估结果的载体，包含总体指标、分组指标与逐样本明细。</p>
 *
 * <h2>报告必须包含的三层信息</h2>
 * <ol>
 *   <li><b>总体指标</b>：一眼判断当前水平</li>
 *   <li><b>分组指标</b>：定位结构性短板（哪类问题最差）</li>
 *   <li><b>逐样本明细</b>：定位具体失败案例（哪个问题、为什么失败）</li>
 * </ol>
 *
 * <p>只有总体指标的报告是没用的——看到 "Faithfulness = 0.72"
 * 之后你仍然不知道该改什么。必须能下钻到具体样本，
 * 看到「q17 检索没召回 embedding 相关内容」，才能采取行动。</p>
 *
 * @author ai-learn
 */
public class EvalReport {

    /** 评估执行时间戳 */
    private long timestamp = System.currentTimeMillis();

    /** 本次评估使用的配置快照，用于归因"指标变化是哪个参数导致的" */
    private Map<String, Object> configSnapshot = new LinkedHashMap<>();

    /** 总体指标 */
    private AggregateMetrics overall;

    /** 分组指标：分类 → 聚合指标 */
    private Map<String, AggregateMetrics> byCategory = new LinkedHashMap<>();

    /** 逐样本明细 */
    private List<SampleResult> samples = new ArrayList<>();

    /** 总耗时 */
    private long totalElapsedMs;

    // ==================== 聚合指标 ====================

    /**
     * 聚合指标容器。
     *
     * <p>检索指标与生成指标分开存储，强调二者是独立维度。</p>
     */
    public static class AggregateMetrics {
        // ---- 检索层 ----
        public double recall;
        public double precision;
        public double mrr;
        public double ndcg;

        // ---- 生成层（RAGAS）----
        public double faithfulness;
        public double answerRelevancy;
        public double contextPrecision;
        public double contextRecall;

        /** 平均耗时（毫秒） */
        public double avgElapsedMs;

        /**
         * 综合分数：检索与生成的加权平均。
         *
         * <p>⚠️ 这个值仅用于快速比较，不应作为决策依据。
         * 加权系数的选取没有理论依据，掩盖了各维度的差异。
         * 真正有用的是看单维度指标的变化。</p>
         */
        public double composite() {
            return (recall + precision + mrr + ndcg
                    + faithfulness + answerRelevancy + contextPrecision + contextRecall) / 8.0;
        }
    }

    /**
     * 单样本评估结果。
     */
    public static class SampleResult {
        public String id;
        public String question;
        public String category;
        public String answer;

        // 检索指标
        public double recall;
        public double precision;
        public double mrr;
        public double ndcg;
        public int retrievedCount;

        // 生成指标
        public double faithfulness;
        public double answerRelevancy;
        public double contextPrecision;
        public double contextRecall;

        public long elapsedMs;

        /** 检索到的分片预览，用于人工核对 */
        public List<String> retrievedPreviews = new ArrayList<>();

        /** 失败原因说明（自动诊断） */
        public String diagnosis;
    }

    // ==================== 报告渲染 ====================

    /**
     * 渲染为人类可读的文本报告。
     */
    public String render() {
        StringBuilder sb = new StringBuilder();

        sb.append("═══════════════════════════════════════════════════════\n");
        sb.append("            RAG 评估报告\n");
        sb.append("═══════════════════════════════════════════════════════\n\n");

        sb.append("样本数: ").append(samples.size())
                .append("    总耗时: ").append(totalElapsedMs).append("ms\n");

        // ---- 配置快照 ----
        if (!configSnapshot.isEmpty()) {
            sb.append("\n【本次评估配置】\n");
            configSnapshot.forEach((k, v) -> sb.append("  ").append(k).append(" = ").append(v).append("\n"));
        }

        // ---- 总体指标 ----
        if (overall != null) {
            sb.append("\n【总体指标】\n");
            sb.append("  ── 检索层（衡量「找得准不准」）──\n");
            sb.append(String.format("    Recall@K     %.3f   %s%n", overall.recall, bar(overall.recall)));
            sb.append(String.format("    Precision@K  %.3f   %s%n", overall.precision, bar(overall.precision)));
            sb.append(String.format("    MRR          %.3f   %s%n", overall.mrr, bar(overall.mrr)));
            sb.append(String.format("    NDCG@K       %.3f   %s%n", overall.ndcg, bar(overall.ndcg)));

            sb.append("  ── 生成层（RAGAS 指标，衡量「答得好不好」）──\n");
            sb.append(String.format("    Faithfulness      %.3f   %s  %s%n",
                    overall.faithfulness, bar(overall.faithfulness), hint("faithfulness", overall.faithfulness)));
            sb.append(String.format("    Answer Relevancy  %.3f   %s  %s%n",
                    overall.answerRelevancy, bar(overall.answerRelevancy), hint("relevancy", overall.answerRelevancy)));
            sb.append(String.format("    Context Precision %.3f   %s%n",
                    overall.contextPrecision, bar(overall.contextPrecision)));
            sb.append(String.format("    Context Recall    %.3f   %s%n",
                    overall.contextRecall, bar(overall.contextRecall)));

            sb.append(String.format("    平均耗时      %.0f ms%n", overall.avgElapsedMs));
            sb.append(String.format("    综合分数      %.3f   （仅供快速比较，勿作决策依据）%n", overall.composite()));
        }

        // ---- 分组指标 ----
        if (!byCategory.isEmpty()) {
            sb.append("\n【分组指标】—— 定位结构性短板\n");
            sb.append(String.format("  %-12s %8s %8s %8s %8s %8s%n",
                    "分类", "Recall", "Faithful", "Relev", "CtxRec", "样本数"));
            for (Map.Entry<String, AggregateMetrics> e : byCategory.entrySet()) {
                AggregateMetrics m = e.getValue();
                long count = samples.stream().filter(s -> e.getKey().equals(s.category)).count();
                sb.append(String.format("  %-12s %8.3f %8.3f %8.3f %8.3f %8d%n",
                        e.getKey(), m.recall, m.faithfulness, m.answerRelevancy, m.contextRecall, count));
            }
        }

        // ---- 失败样本明细 ----
        List<SampleResult> failures = samples.stream()
                .filter(s -> s.diagnosis != null && !s.diagnosis.isBlank())
                .toList();

        if (!failures.isEmpty()) {
            sb.append("\n【需要注意的样本】—— 共 ").append(failures.size()).append(" 条\n");
            for (SampleResult s : failures) {
                sb.append("\n  [").append(s.id).append("] ").append(s.question).append("\n");
                sb.append("      分类: ").append(s.category)
                        .append(" | 召回数: ").append(s.retrievedCount).append("\n");
                sb.append("      Recall=").append(String.format("%.2f", s.recall))
                        .append(" Faithfulness=").append(String.format("%.2f", s.faithfulness))
                        .append(" Relevancy=").append(String.format("%.2f", s.answerRelevancy))
                        .append("\n");
                sb.append("      诊断: ").append(s.diagnosis).append("\n");
                sb.append("      回答: ").append(truncate(s.answer, 150)).append("\n");
                if (!s.retrievedPreviews.isEmpty()) {
                    sb.append("      召回片段:\n");
                    for (int i = 0; i < Math.min(3, s.retrievedPreviews.size()); i++) {
                        sb.append("        #").append(i + 1).append(" ")
                                .append(truncate(s.retrievedPreviews.get(i), 100)).append("\n");
                    }
                }
            }
        } else {
            sb.append("\n【需要注意的样本】无 —— 所有样本指标均达标\n");
        }

        sb.append("\n═══════════════════════════════════════════════════════\n");
        sb.append("解读提示：\n");
        sb.append("  · Recall 低 → 检索没找到正确内容，优先调整切分与召回策略\n");
        sb.append("  · Faithfulness 低 → 模型在编造，检查 Prompt 约束与上下文质量\n");
        sb.append("  · Answer Relevancy 低 → 答非所问，检查 Prompt 与问题理解环节\n");
        sb.append("  · Context Precision 低 → 排序有问题，考虑加强重排序\n");
        sb.append("  · Context Recall 低 → 覆盖度不足，考虑多查询扩展与 HyDE\n");
        sb.append("═══════════════════════════════════════════════════════\n");

        return sb.toString();
    }

    /**
     * 生成简易进度条，让数值对比更直观。
     */
    private String bar(double value) {
        int filled = (int) Math.round(Math.max(0, Math.min(1, value)) * 20);
        return "█".repeat(filled) + "░".repeat(20 - filled);
    }

    /**
     * 针对低分指标给出具体的改进建议。
     */
    private String hint(String metric, double value) {
        if (value >= 0.8) {
            return "";
        }
        return switch (metric) {
            case "faithfulness" -> "← 偏低：加强 Prompt 的「禁止编造」约束，或提高检索精度";
            case "relevancy" -> "← 偏低：检查 Prompt 是否要求直接回答问题";
            default -> "";
        };
    }

    private String truncate(String s, int max) {
        if (s == null) {
            return "(空)";
        }
        String flat = s.replaceAll("\\s+", " ").trim();
        return flat.length() <= max ? flat : flat.substring(0, max) + "...";
    }

    // ==================== getter / setter ====================

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public Map<String, Object> getConfigSnapshot() {
        return configSnapshot;
    }

    public void setConfigSnapshot(Map<String, Object> configSnapshot) {
        this.configSnapshot = configSnapshot;
    }

    public AggregateMetrics getOverall() {
        return overall;
    }

    public void setOverall(AggregateMetrics overall) {
        this.overall = overall;
    }

    public Map<String, AggregateMetrics> getByCategory() {
        return byCategory;
    }

    public void setByCategory(Map<String, AggregateMetrics> byCategory) {
        this.byCategory = byCategory;
    }

    public List<SampleResult> getSamples() {
        return samples;
    }

    public void setSamples(List<SampleResult> samples) {
        this.samples = samples;
    }

    public long getTotalElapsedMs() {
        return totalElapsedMs;
    }

    public void setTotalElapsedMs(long totalElapsedMs) {
        this.totalElapsedMs = totalElapsedMs;
    }
}
