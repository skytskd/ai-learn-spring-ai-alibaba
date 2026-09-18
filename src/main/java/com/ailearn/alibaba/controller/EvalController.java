package com.ailearn.alibaba.controller;

import com.ailearn.alibaba.eval.EvalReport;
import com.ailearn.alibaba.eval.EvalService;
import com.ailearn.alibaba.eval.GoldenTestSet;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * <h1>EvalController — RAG 评估接口</h1>
 *
 * <p>把评估体系暴露为 HTTP 接口，使它可以被 CI、定时任务或
 * 前端页面调用。</p>
 *
 * <h2>接入 CI 的建议做法</h2>
 * <pre>
 *   # 在 CI 流程中加入评估门禁
 *   curl -X POST http://localhost:8080/api/eval/run?limit=20
 *
 *   # 解析返回的 overall.faithfulness，低于阈值则构建失败
 * </pre>
 *
 * <h2>⚠️ 成本提醒</h2>
 * <p>一次完整评估（20 样本）约消耗 120~180 次 LLM 调用。
 * 不要放进「每次提交触发」的 CI，应用定时任务或手动触发。</p>
 *
 * @author ai-learn
 */
@Slf4j
@RestController
@RequestMapping("/api/eval")
@RequiredArgsConstructor
public class EvalController {

    private final EvalService evalService;
    private final GoldenTestSet goldenTestSet;

    /**
     * 查看测试集内容（不触发评估，零成本）。
     *
     * <p>调优的第一步应该是看清楚测试集问的是什么——
     * 否则你无法判断指标是否可信。</p>
     */
    @GetMapping("/dataset")
    public Map<String, Object> dataset() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("size", goldenTestSet.size());
        result.put("categories", goldenTestSet.categories());
        result.put("samples", goldenTestSet.all().stream().map(s -> {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("id", s.getId());
            item.put("category", s.getCategory());
            item.put("question", s.getQuestion());
            item.put("groundTruth", s.getGroundTruthAnswer());
            item.put("keyPhrases", s.getKeyPhrases());
            return item;
        }).toList());
        return result;
    }

    /**
     * 执行完整评估。
     *
     * @param limit 限制样本数，用于快速验证（默认全部）
     */
    @PostMapping("/run")
    public Map<String, Object> run(@RequestParam(required = false) Integer limit) {
        log.info("[评估接口] 触发评估，limit={}", limit);

        EvalReport report = evalService.run(limit);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("sampleCount", report.getSamples().size());
        result.put("totalElapsedMs", report.getTotalElapsedMs());
        result.put("configSnapshot", report.getConfigSnapshot());

        EvalReport.AggregateMetrics m = report.getOverall();
        Map<String, Object> overall = new LinkedHashMap<>();
        overall.put("retrieval", Map.of(
                "recall", round(m.recall),
                "precision", round(m.precision),
                "mrr", round(m.mrr),
                "ndcg", round(m.ndcg)));
        overall.put("generation", Map.of(
                "faithfulness", round(m.faithfulness),
                "answerRelevancy", round(m.answerRelevancy),
                "contextPrecision", round(m.contextPrecision),
                "contextRecall", round(m.contextRecall)));
        overall.put("avgElapsedMs", Math.round(m.avgElapsedMs));
        overall.put("composite", round(m.composite()));
        result.put("overall", overall);

        // 分组指标
        Map<String, Object> byCategory = new LinkedHashMap<>();
        report.getByCategory().forEach((cat, am) -> byCategory.put(cat, Map.of(
                "recall", round(am.recall),
                "faithfulness", round(am.faithfulness),
                "answerRelevancy", round(am.answerRelevancy),
                "contextRecall", round(am.contextRecall))));
        result.put("byCategory", byCategory);

        // 问题样本
        List<Map<String, Object>> issues = report.getSamples().stream()
                .filter(s -> s.diagnosis != null && !s.diagnosis.isBlank())
                .map(s -> {
                    Map<String, Object> item = new LinkedHashMap<>();
                    item.put("id", s.id);
                    item.put("question", s.question);
                    item.put("diagnosis", s.diagnosis);
                    return item;
                })
                .toList();
        result.put("issues", issues);
        result.put("issueCount", issues.size());

        return result;
    }

    /**
     * 获取文本格式的完整报告，便于直接阅读或存档。
     */
    @PostMapping(value = "/report", produces = "text/plain;charset=UTF-8")
    public String report(@RequestParam(required = false) Integer limit) {
        log.info("[评估接口] 生成文本报告，limit={}", limit);
        return evalService.run(limit).render();
    }

    /**
     * 快速冒烟测试：只跑 3 个样本。
     *
     * <p>用于验证评估链路本身是否通畅，成本极低。</p>
     */
    @PostMapping("/smoke")
    public Map<String, Object> smoke() {
        log.info("[评估接口] 冒烟测试");
        return run(3);
    }

    private double round(double v) {
        return Math.round(v * 1000) / 1000.0;
    }
}
