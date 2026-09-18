package com.ailearn.alibaba.controller;

import com.ailearn.alibaba.config.RagProperties;
import com.ailearn.alibaba.model.ChatRequest;
import com.ailearn.alibaba.service.RagService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * <h1>RagController（强化版）— RAG 检索增强生成接口</h1>
 *
 * <h2>接口列表</h2>
 * <table border="1">
 *   <tr><th>接口</th><th>说明</th></tr>
 *   <tr><td>POST /api/rag/ask</td>
 *       <td>强化版问答：查询改写 + 混合检索 + 重排序</td></tr>
 *   <tr><td>POST /api/rag/ask-baseline</td>
 *       <td>基础版问答：纯向量检索，用于对比</td></tr>
 *   <tr><td>POST /api/rag/explain</td>
 *       <td>检索诊断：展示每一路召回了什么、排名如何变化</td></tr>
 *   <tr><td>GET  /api/rag/config</td>
 *       <td>查看当前 RAG 参数配置</td></tr>
 *   <tr><td>POST /api/rag/rebuild</td>
 *       <td>重建索引（补充 API 额度后恢复混合检索，无需重启）</td></tr>
 * </table>
 *
 * <h2>怎么做对比实验？</h2>
 * <p>同一个问题分别调用 {@code /ask} 与 {@code /ask-baseline}，
 * 再用 {@code /explain} 看中间过程。这是理解每个优化环节
 * 到底带来多少收益的最直接方式——比看任何教程都有效。</p>
 *
 * @author ai-learn
 */
@Slf4j
@RestController
@RequestMapping("/api/rag")
@RequiredArgsConstructor
public class RagController {

    private final RagService ragService;
    private final RagProperties ragProperties;

    /**
     * 强化版 RAG 问答。
     *
     * <p>完整流水线：查询改写 → 稠密+稀疏双路召回 → RRF 融合 → 重排序 → 生成。</p>
     */
    @PostMapping("/ask")
    public Map<String, String> ask(@RequestBody ChatRequest request) {
        log.info("[RAG问答-强化版] {}", request.getMessage());
        String answer = ragService.ask(request.getMessage());
        return Map.of("content", answer);
    }

    /**
     * 基础版 RAG 问答（对照用）。
     *
     * <p>只有向量检索，无融合、无重排。把它与强化版对比，
     * 能直观感受「朴素 RAG」与「进阶 RAG」的差距。</p>
     */
    @PostMapping("/ask-baseline")
    public Map<String, String> askBaseline(@RequestBody ChatRequest request) {
        log.info("[RAG问答-基础版] {}", request.getMessage());
        String answer = ragService.askBaseline(request.getMessage());
        return Map.of("content", answer);
    }

    /**
     * 检索诊断：完整展示检索流水线的每一环。
     *
     * <h3>返回内容</h3>
     * <ul>
     *   <li>{@code denseHits} — 向量检索召回结果</li>
     *   <li>{@code sparseHits} — BM25 检索召回结果</li>
     *   <li>{@code fusedHits} — RRF 融合后的排序</li>
     *   <li>{@code rerankedHits} — 重排序后的最终结果</li>
     * </ul>
     *
     * <p>调优 RAG 时最有用的接口。当回答不对时，
     * 先看这个接口：是某一路没召回？还是融合把好结果压下去了？
     * 还是重排排错了？—— 一眼就能归因。</p>
     */
    @PostMapping("/explain")
    public Map<String, Object> explain(@RequestBody ChatRequest request) {
        log.info("[RAG诊断] {}", request.getMessage());
        return ragService.explain(request.getMessage());
    }

    /**
     * 查看当前 RAG 配置参数。
     */
    @GetMapping("/config")
    public Map<String, Object> config() {
        Map<String, Object> cfg = new LinkedHashMap<>();
        cfg.put("chunkSize", ragProperties.getChunkSize());
        cfg.put("minChunkSizeChars", ragProperties.getMinChunkSizeChars());
        cfg.put("denseTopK", ragProperties.getDenseTopK());
        cfg.put("sparseTopK", ragProperties.getSparseTopK());
        cfg.put("fusionTopK", ragProperties.getFusionTopK());
        cfg.put("rerankTopN", ragProperties.getRerankTopN());
        cfg.put("similarityThreshold", ragProperties.getSimilarityThreshold());
        cfg.put("enableRerank", ragProperties.isEnableRerank());
        cfg.put("enableQueryExpansion", ragProperties.isEnableQueryExpansion());
        cfg.put("queryExpansionCount", ragProperties.getQueryExpansionCount());
        cfg.put("enableHyde", ragProperties.isEnableHyde());

        Map<String, Object> knowledgeBase = new LinkedHashMap<>();
        knowledgeBase.put("ready", ragService.isReady());
        knowledgeBase.put("chunkCount", ragService.getAllChunks().size());
        // 向量索引状态：BM25 索引随应用启动即可用，向量索引为惰性构建
        knowledgeBase.put("vectorIndexReady", ragService.isVectorIndexReady());
        knowledgeBase.put("vectorIndexError", ragService.getVectorIndexError());
        cfg.put("knowledgeBase", knowledgeBase);

        return cfg;
    }

    /**
     * 重建索引（含向量索引）。
     *
     * <p>使用场景：应用启动时 DashScope 配额耗尽 / 网络异常，
     * 导致向量索引未能建立、混合检索降级为纯 BM25。
     * 补充额度后<b>无需重启应用</b>，调用本接口即可恢复混合检索。</p>
     */
    @PostMapping("/rebuild")
    public Map<String, Object> rebuild() {
        log.info("[RAG] 收到索引重建请求");
        return ragService.rebuild();
    }
}
