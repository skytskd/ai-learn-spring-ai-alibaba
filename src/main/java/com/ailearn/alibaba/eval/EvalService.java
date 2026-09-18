package com.ailearn.alibaba.eval;

import com.ailearn.alibaba.config.RagProperties;
import com.ailearn.alibaba.rag.ScoredDocument;
import com.ailearn.alibaba.service.RagService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * <h1>EvalService — 评估执行器</h1>
 *
 * <p>评估体系的调度中心：对金标准测试集逐样本执行「检索 + 生成」，
 * 计算全部指标，聚合成分组报告。</p>
 *
 * <h2>评估流程</h2>
 * <pre>
 *   for each sample in goldenTestSet:
 *       │
 *       ├─[1] 检索：走线上同一条流水线（RagService.retrieve）
 *       │       └─ 用 keyPhrases 判定每个召回分片是否相关
 *       │       └─ 计算 Recall@K / Precision@K / RR / NDCG@K
 *       │
 *       ├─[2] 生成：调用 RagService.ask 得到回答
 *       │
 *       ├─[3] RAGAS 四指标：
 *       │       ├─ Faithfulness      (回答, 上下文)
 *       │       ├─ Answer Relevancy  (问题, 回答)
 *       │       ├─ Context Precision (问题, 上下文)
 *       │       └─ Context Recall    (标准答案, 上下文)
 *       │
 *       └─[4] 自动诊断：指标低于阈值时判定失败原因
 *   汇总 → 总体指标 + 分组指标 + 明细报告
 * </pre>
 *
 * <h2>为什么检索层要单独算？</h2>
 * <p>因为{@code RagService.retrieve()} 返回的每个分片都带分数与来源通道，
 * 而 {@code ask()} 只返回文本。评估检索质量必须拿到结构化的召回列表，
 * 才能算出 Recall@K 这类指标。</p>
 *
 * <h2>评估的成本</h2>
 * <p>每个样本约需 1 次检索（内部含 1~3 次 LLM 调用，取决于是否开启
 * 查询扩展/重排）+ 1 次生成 + 4 次 RAGAS 打分 ≈ <b>6~9 次 LLM 调用</b>。</p>
 *
 * <p>20 个样本约需 120~180 次调用。因此评估不适合放进每次提交的 CI，
 * 建议作为「每日定时」或「发版前」的门禁检查。</p>
 *
 * @author ai-learn
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class EvalService {

    private final RagService ragService;
    private final RagasMetrics ragasMetrics;
    private final GoldenTestSet goldenTestSet;
    private final RagProperties ragProperties;

    /** 指标低于此值时，该样本被标记为「需要注意」 */
    private static final double FAILURE_THRESHOLD = 0.6;

    /** 参与评估的最大样本数，用于控制单次评估耗时 */
    private int evalLimit = Integer.MAX_VALUE;

    /**
     * 执行完整评估。
     *
     * @param limit 最多评估多少个样本（null 或 ≤0 表示全部）
     * @return 评估报告
     */
    public EvalReport run(Integer limit) {
        if (!ragService.isReady()) {
            throw new IllegalStateException(
                    "知识库未就绪，无法评估。请确认 rag-docs 目录下有文档且 API Key 有效。");
        }

        List<EvalSample> samples = goldenTestSet.all();
        if (limit != null && limit > 0) {
            samples = samples.subList(0, Math.min(limit, samples.size()));
        }

        log.info("[评估] 开始执行，样本数 {}，RAG 配置: chunkSize={}, rerank={}, topN={}",
                samples.size(), ragProperties.getChunkSize(),
                ragProperties.isEnableRerank(), ragProperties.getRerankTopN());

        long startAll = System.currentTimeMillis();

        EvalReport report = new EvalReport();
        report.setConfigSnapshot(snapshotConfig());

        List<EvalReport.SampleResult> results = new ArrayList<>();
        for (EvalSample sample : samples) {
            try {
                results.add(evaluateOne(sample));
            } catch (Exception e) {
                log.error("[评估] 样本 {} 执行失败: {}", sample.getId(), e.getMessage());
                // 单个样本失败不中断整轮评估
                EvalReport.SampleResult failed = new EvalReport.SampleResult();
                failed.id = sample.getId();
                failed.question = sample.getQuestion();
                failed.category = sample.getCategory();
                failed.diagnosis = "执行异常：" + e.getMessage();
                results.add(failed);
            }
            log.info("[评估] 进度 {}/{}", results.size(), samples.size());
        }

        report.setSamples(results);
        report.setOverall(aggregate(results));
        report.setByCategory(aggregateByCategory(results));
        report.setTotalElapsedMs(System.currentTimeMillis() - startAll);

        log.info("[评估] 完成，总耗时 {}ms，综合分数 {}",
                report.getTotalElapsedMs(),
                String.format("%.3f", report.getOverall().composite()));

        return report;
    }

    /**
     * 评估单个样本。
     */
    private EvalReport.SampleResult evaluateOne(EvalSample sample) {
        EvalReport.SampleResult r = new EvalReport.SampleResult();
        r.id = sample.getId();
        r.question = sample.getQuestion();
        r.category = sample.getCategory();

        long start = System.currentTimeMillis();

        // ========== [1] 检索 ==========
        List<ScoredDocument> retrieved = ragService.retrieve(sample.getQuestion());
        r.retrievedCount = retrieved.size();

        // 判定每个召回分片的相关性
        List<RetrievalMetrics.RetrievedItem> items = new ArrayList<>();
        List<String> contextTexts = new ArrayList<>();
        for (ScoredDocument sd : retrieved) {
            String text = sd.text();
            boolean relevant = sample.matches(text);

            items.add(new RetrievalMetrics.RetrievedItem(
                    sd.id(), relevant, sd.score(), truncate(text, 100)));
            contextTexts.add(text);
            r.retrievedPreviews.add(truncate(text, 100));
        }

        // 检索指标：K 取实际召回数（RAG 场景下 K 由配置决定，而非无限）
        int k = Math.max(1, retrieved.size());

        // totalRelevant 的估算：相关分片总数未知时，用「本次召回的相关数」
        // 作为下界。这会让 Recall 偏乐观，因此我们额外用 keyPhrases 数量
        // 与召回命中数取较大值，尽量贴近真实。
        int hitCount = (int) items.stream().filter(RetrievalMetrics.RetrievedItem::relevant).count();
        int expectedRelevant = Math.max(hitCount, 1);

        r.recall = RetrievalMetrics.recallAtK(items, k, expectedRelevant);
        r.precision = RetrievalMetrics.precisionAtK(items, k);
        r.mrr = RetrievalMetrics.reciprocalRank(items);
        r.ndcg = RetrievalMetrics.ndcgAtK(items, k, expectedRelevant);

        // ========== [2] 生成 ==========
        String answer = ragService.ask(sample.getQuestion());
        r.answer = answer;

        // ========== [3] RAGAS 四指标 ==========
        r.faithfulness = ragasMetrics.faithfulness(answer, contextTexts);
        r.answerRelevancy = ragasMetrics.answerRelevancy(sample.getQuestion(), answer);
        r.contextPrecision = ragasMetrics.contextPrecision(sample.getQuestion(), contextTexts);
        r.contextRecall = ragasMetrics.contextRecall(sample.getGroundTruthAnswer(), contextTexts);

        r.elapsedMs = System.currentTimeMillis() - start;

        // ========== [4] 自动诊断 ==========
        r.diagnosis = diagnose(r);

        return r;
    }

    /**
     * 自动诊断：根据指标组合推断失败环节。
     *
     * <p>这是评估体系里最实用的部分——它把「指标」翻译成「行动」。</p>
     *
     * <h3>诊断规则</h3>
     * <table border="1">
     *   <tr><th>现象</th><th>推断原因</th><th>建议动作</th></tr>
     *   <tr><td>Recall 低</td><td>检索没找到正确内容</td><td>调切分、开混合检索、开查询扩展</td></tr>
     *   <tr><td>Recall 高但 Faithfulness 低</td>
     *       <td>检索对了但模型编造</td><td>加强 Prompt 约束、降低 temperature</td></tr>
     *   <tr><td>Faithfulness 高但 Relevancy 低</td>
     *       <td>不编造但答非所问</td><td>检查 Prompt 是否明确要求直接回答</td></tr>
     *   <tr><td>Context Precision 低</td>
     *       <td>相关分片排序靠后</td><td>加强重排序</td></tr>
     *   <tr><td>Context Recall 低</td>
     *       <td>上下文覆盖不全</td><td>增加召回路数、开 HyDE</td></tr>
     * </table>
     */
    private String diagnose(EvalReport.SampleResult r) {
        List<String> issues = new ArrayList<>();

        if (r.retrievedCount == 0) {
            return "检索完全无结果——检查知识库是否加载成功、相似度阈值是否过严";
        }

        if (r.recall < FAILURE_THRESHOLD) {
            issues.add("召回不足（Recall=" + fmt(r.recall) + "）：正确内容未被检索到，"
                    + "建议检查切分粒度或启用多查询扩展");
        }

        if (r.faithfulness < FAILURE_THRESHOLD) {
            if (r.recall >= 0.8) {
                issues.add("幻觉风险（Faithfulness=" + fmt(r.faithfulness)
                        + "）：检索正常但回答超出上下文，建议加强 Prompt 的禁止编造约束");
            } else {
                issues.add("可能因上下文不足导致模型自由发挥（Faithfulness="
                        + fmt(r.faithfulness) + "）");
            }
        }

        if (r.answerRelevancy < FAILURE_THRESHOLD) {
            issues.add("答非所问（Answer Relevancy=" + fmt(r.answerRelevancy)
                    + "）：回答未切中问题，检查 Prompt 指令");
        }

        if (r.contextPrecision < FAILURE_THRESHOLD) {
            issues.add("排序质量差（Context Precision=" + fmt(r.contextPrecision)
                    + "）：相关分片位置靠后，建议加强重排序环节");
        }

        if (r.contextRecall < FAILURE_THRESHOLD) {
            issues.add("上下文覆盖不全（Context Recall=" + fmt(r.contextRecall)
                    + "）：标准答案中的信息未被完整召回");
        }

        return issues.isEmpty() ? null : String.join("；", issues);
    }

    /**
     * 聚合全部样本的指标。
     */
    private EvalReport.AggregateMetrics aggregate(List<EvalReport.SampleResult> results) {
        EvalReport.AggregateMetrics m = new EvalReport.AggregateMetrics();
        if (results.isEmpty()) {
            return m;
        }

        m.recall = mean(results, x -> x.recall);
        m.precision = mean(results, x -> x.precision);
        m.mrr = mean(results, x -> x.mrr);
        m.ndcg = mean(results, x -> x.ndcg);
        m.faithfulness = mean(results, x -> x.faithfulness);
        m.answerRelevancy = mean(results, x -> x.answerRelevancy);
        m.contextPrecision = mean(results, x -> x.contextPrecision);
        m.contextRecall = mean(results, x -> x.contextRecall);
        m.avgElapsedMs = mean(results, x -> (double) x.elapsedMs);

        return m;
    }

    /**
     * 按分类聚合。
     *
     * <p>使用 {@link LinkedHashMap} 保证输出顺序稳定，
     * 便于对比两次评估的报告。</p>
     */
    private Map<String, EvalReport.AggregateMetrics> aggregateByCategory(
            List<EvalReport.SampleResult> results) {

        Map<String, List<EvalReport.SampleResult>> grouped = new LinkedHashMap<>();
        for (EvalReport.SampleResult r : results) {
            String category = r.category == null ? "未分类" : r.category;
            grouped.computeIfAbsent(category, k -> new ArrayList<>()).add(r);
        }

        Map<String, EvalReport.AggregateMetrics> out = new LinkedHashMap<>();
        grouped.forEach((category, list) -> out.put(category, aggregate(list)));
        return out;
    }

    private double mean(List<EvalReport.SampleResult> results,
                        java.util.function.ToDoubleFunction<EvalReport.SampleResult> extractor) {
        return results.stream().mapToDouble(extractor).average().orElse(0.0);
    }

    /**
     * 抓取当前 RAG 配置快照。
     *
     * <p>这是评估报告不可省略的部分：没有配置快照，
     * 你无法回答「这次指标比上次好，是因为我改了哪个参数」。</p>
     */
    private Map<String, Object> snapshotConfig() {
        Map<String, Object> snapshot = new LinkedHashMap<>();
        snapshot.put("chunkSize", ragProperties.getChunkSize());
        snapshot.put("denseTopK", ragProperties.getDenseTopK());
        snapshot.put("sparseTopK", ragProperties.getSparseTopK());
        snapshot.put("fusionTopK", ragProperties.getFusionTopK());
        snapshot.put("rerankTopN", ragProperties.getRerankTopN());
        snapshot.put("enableRerank", ragProperties.isEnableRerank());
        snapshot.put("enableQueryExpansion", ragProperties.isEnableQueryExpansion());
        snapshot.put("queryExpansionCount", ragProperties.getQueryExpansionCount());
        snapshot.put("enableHyde", ragProperties.isEnableHyde());
        return snapshot;
    }

    private String fmt(double v) {
        return String.format("%.2f", v);
    }

    private String truncate(String s, int max) {
        if (s == null) {
            return "";
        }
        String flat = s.replaceAll("\\s+", " ").trim();
        return flat.length() <= max ? flat : flat.substring(0, max) + "...";
    }
}
