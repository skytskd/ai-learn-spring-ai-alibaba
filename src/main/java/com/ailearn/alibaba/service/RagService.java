package com.ailearn.alibaba.service;

import com.ailearn.alibaba.config.RagProperties;
import com.ailearn.alibaba.rag.Bm25SparseRetriever;
import com.ailearn.alibaba.rag.LlmReranker;
import com.ailearn.alibaba.rag.QueryRewriter;
import com.ailearn.alibaba.rag.RrfFusion;
import com.ailearn.alibaba.rag.ScoredDocument;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * <h1>第4课（强化版）：RagService — 工业级 RAG 检索增强生成</h1>
 *
 * <p>本项目最初版本演示的是「Naive RAG」：加载文档 → 向量化 → 相似度检索 → 拼接 Prompt。
 * 该版本用于理解概念足够，但<b>在真实场景中召回率与准确率都不达标</b>。</p>
 *
 * <p>本类实现了进阶 RAG 的完整流水线，对应业界验证过的四层优化。</p>
 *
 * <h2>完整流水线</h2>
 * <pre>
 *   用户问题
 *      │
 *      ├─[1] 查询改写（QueryRewriter）
 *      │      ├─ 多查询扩展：生成 N 个语义变体，分别检索
 *      │      └─ HyDE：生成假想答案文档，用其向量检索
 *      │
 *      ├─[2] 多路召回（Hybrid Retrieval）
 *      │      ├─ 稠密检索：向量相似度，擅长语义匹配
 *      │      └─ 稀疏检索：BM25 词频，擅长专有名词 / 编号 / 代码
 *      │
 *      ├─[3] 排名融合（RRF — Reciprocal Rank Fusion）
 *      │      └─ 只看排名不看分数，免调参、抗异常、奖励双路共识
 *      │
 *      └─[4] 重排序（Rerank）
 *             └─ Cross-Encoder 精排，Top-20 → Top-5
 *                    │
 *                    ▼
 *              拼接上下文 → 调用 LLM → 带引用的回答
 * </pre>
 *
 * <h2>为什么要这样设计？</h2>
 * <p>信息检索的黄金法则是<b>「召回求全，精排求准」</b>：</p>
 * <ul>
 *   <li><b>召回阶段</b>要在海量文档中尽可能不漏掉任何相关内容——
 *       因此用多查询 + 双通道，宁滥勿缺</li>
 *   <li><b>精排阶段</b>要在候选中挑出最相关的几条送给 LLM——
 *       因为 LLM 上下文窗口有限，塞入无关内容会稀释注意力、诱发幻觉</li>
 * </ul>
 *
 * <p>把全部候选直接塞给 LLM（很多简化实现的做法）看似省事，
 * 实则是幻觉的主要来源之一。</p>
 *
 * @author ai-learn
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RagService {

    private final ChatClient chatClient;
    private final EmbeddingModel embeddingModel;
    private final Bm25SparseRetriever sparseRetriever;
    private final RrfFusion rrfFusion;
    private final LlmReranker reranker;
    private final QueryRewriter queryRewriter;
    private final RagProperties ragProperties;

    /** 向量存储是否已构建（延迟构建，见 ensureVectorIndex） */
    private volatile boolean vectorIndexReady = false;

    /** 向量索引构建失败的原因，供 /config 与 /explain 返回，便于排障 */
    private volatile String vectorIndexError = null;

    /**
     * <p><b>为什么用 volatile 标记而非直接缓存对象？</b>
     * 应用启动不应依赖外部 AI 服务可用性。若在 {@code @PostConstruct}
     * 中同步调用 Embedding 接口，一旦配额耗尽 / 网络不通，整个应用直接起不来。
     * 因此把向量索引改为<b>首次使用时惰性构建</b>：启动只做本地切分 + BM25 索引
     * （纯 CPU、零外部依赖），向量索引在第一次检索时再建。</p>
     */
    private volatile VectorStore vectorStore;

    /** 全部文档分片，供评估模块复用 */
    private List<Document> allChunks = List.of();

    // ==================== 初始化 ====================

    /**
     * 加载文档、切分、建立 BM25 索引。
     *
     * <p>本方法<b>不做任何网络调用</b>，保证应用启动不被外部服务拖累。
     * 向量索引由 {@link #ensureVectorIndex()} 在首次检索时惰性构建。</p>
     */
    @PostConstruct
    public void init() {
        log.info("[RAG] 开始构建知识库...");

        try {
            List<Document> rawDocs = loadRawDocuments();
            if (rawDocs.isEmpty()) {
                log.warn("[RAG] 未找到任何文档，RAG 功能不可用");
                return;
            }

            // ---------- 文档切分（Chunking）----------
            // 切分策略是 RAG 里最"脏"也最关键的活：
            //   - 切太大：单块混入多个主题，向量表示被"平均"掉，检索失准
            //   - 切太小：语义不完整，模型拿不到足够上下文
            // TokenTextSplitter 按 token 数切分并保留重叠区，
            // 重叠的意义是防止关键句正好被切在边界上而两边都不完整。
            TokenTextSplitter splitter = TokenTextSplitter.builder()
                    .withChunkSize(ragProperties.getChunkSize())
                    .withMinChunkSizeChars(ragProperties.getMinChunkSizeChars())
                    .withMinChunkLengthToEmbed(ragProperties.getMinChunkLengthToEmbed())
                    .withMaxNumChunks(ragProperties.getMaxNumChunks())
                    // keepSeparator=true 保留分隔符，避免把换行吃掉导致语义粘连
                    .withKeepSeparator(true)
                    .build();

            allChunks = splitter.apply(rawDocs);
            log.info("[RAG] 文档切分完成：{} 篇原文 → {} 个分片（chunk-size={}, 重叠保留）",
                    rawDocs.size(), allChunks.size(), ragProperties.getChunkSize());

            // ---------- 索引 1：BM25 倒排索引（稀疏检索，纯本地）----------
            sparseRetriever.build(allChunks);
            log.info("[RAG] BM25 倒排索引构建完成");

            // ---------- 索引 2：向量索引（稠密检索，惰性构建）----------
            // 仅在前置条件满足时尝试构建，失败也不影响应用启动：
            //   BM25 单路检索仍可用，混合检索自动降级为稀疏检索
            if (isEmbeddingConfigured()) {
                ensureVectorIndex();
            } else {
                log.warn("[RAG] 未检测到有效的 DashScope API Key，跳过向量索引构建。"
                        + "BM25 稀疏检索仍可用；配置密钥后调用 /api/rag/rebuild 即可启用混合检索。");
            }

            log.info("[RAG] 知识库就绪（BM25 已启用，向量索引={}），分片数 {}",
                    vectorIndexReady ? "已就绪" : "未就绪", allChunks.size());

        } catch (Exception e) {
            log.error("[RAG] 知识库初始化失败", e);
        }
    }

    /**
     * 判断是否配置了可用的 API Key。
     *
     * <p>占位符 {@code your-api-key-here} 视为未配置——
     * 否则会拿着一串无效字符去打远程接口，白白浪费一次网络往返和重试。</p>
     */
    private boolean isEmbeddingConfigured() {
        String key = System.getenv("AI_API_KEY");
        if (key == null || key.isBlank()) {
            key = System.getProperty("AI_API_KEY");
        }
        if (key == null || key.isBlank()) {
            // 环境变量缺失时，回退检查 application.yml 里是否仍是占位符
            return false;
        }
        return !key.contains("your-api-key-here");
    }

    /**
     * 惰性构建向量索引（幂等，线程安全）。
     *
     * <p>使用双重检查锁：构建过程需要调用 Embedding 接口，可能耗时数秒，
     * 不能每次都重建；同时要防止并发首访时重复构建。</p>
     *
     * @return 向量索引是否可用
     */
    private boolean ensureVectorIndex() {
        if (vectorIndexReady) {
            return true;
        }
        synchronized (this) {
            if (vectorIndexReady) {
                return true;
            }
            try {
                VectorStore store = SimpleVectorStore.builder(embeddingModel).build();
                store.add(allChunks);
                this.vectorStore = store;
                this.vectorIndexReady = true;
                this.vectorIndexError = null;
                log.info("[RAG] 向量索引构建完成：{} 个向量", allChunks.size());
                return true;
            } catch (Exception e) {
                this.vectorIndexError = e.getMessage();
                log.warn("[RAG] 向量索引构建失败，降级为纯 BM25 检索：{}", e.getMessage());
                return false;
            }
        }
    }

    /**
     * 强制重建索引（含向量索引）。
     *
     * <p>用途：补充 API Key 额度后无需重启应用即可恢复混合检索。</p>
     *
     * @return 重建结果，供接口直接返回
     */
    public Map<String, Object> rebuild() {
        Map<String, Object> result = new LinkedHashMap<>();
        synchronized (this) {
            vectorIndexReady = false;
            vectorStore = null;
            vectorIndexError = null;
        }
        boolean ok = ensureVectorIndex();
        result.put("vectorIndexReady", ok);
        result.put("chunkCount", allChunks.size());
        result.put("error", vectorIndexError);
        return result;
    }

    /**
     * 从 classpath:/rag-docs/ 加载全部文本文件。
     *
     * <p>使用 {@link PathMatchingResourcePatternResolver} 通配加载，
     * 新增文档时<b>无需修改代码</b>——这是相比硬编码文件名的重要改进。</p>
     */
    private List<Document> loadRawDocuments() throws Exception {
        List<Document> docs = new ArrayList<>();

        PathMatchingResourcePatternResolver resolver = new PathMatchingResourcePatternResolver();
        Resource[] resources = resolver.getResources("classpath*:/rag-docs/*.txt");

        for (Resource resource : resources) {
            if (!resource.exists()) {
                continue;
            }
            List<Document> loaded = new TextReader(resource).get();

            // 打上来源标记，便于最终回答里做引用溯源
            String filename = resource.getFilename();
            for (Document doc : loaded) {
                doc.getMetadata().put("source", filename);
            }

            int chars = loaded.stream().mapToInt(d -> d.getText().length()).sum();
            log.info("[RAG] 加载文档 {}：{} 字符", filename, chars);
            docs.addAll(loaded);
        }

        return docs;
    }

    // ==================== 检索核心 ====================

    /**
     * 完整检索流水线：查询改写 → 多路召回 → RRF 融合 → 重排序。
     *
     * <p>本方法是 RAG 调优的核心，评估模块也直接复用它，
     * 保证「线上检索」与「离线评估」走的是同一条链路——
     * 否则评估结果无法反映真实表现。</p>
     *
     * @param question 用户问题
     * @return 精排后的 Top-N 文档，已按相关性降序
     */
    public List<ScoredDocument> retrieve(String question) {
        if (allChunks.isEmpty()) {
            log.warn("[RAG] 知识库为空，无法检索");
            return List.of();
        }

        // 惰性构建向量索引：首次检索时建立，失败则自动降级为纯 BM25
        boolean denseAvailable = ensureVectorIndex();

        long start = System.currentTimeMillis();

        // ---------- [1] 查询改写 ----------
        List<String> queries = new ArrayList<>();
        queries.add(question);

        if (ragProperties.isEnableQueryExpansion()) {
            queries.addAll(queryRewriter.expand(question, ragProperties.getQueryExpansionCount())
                    .stream()
                    .filter(q -> !q.equals(question))
                    .toList());
        }

        if (ragProperties.isEnableHyde()) {
            String hypothetical = queryRewriter.hypothesize(question);
            if (!hypothetical.equals(question)) {
                queries.add(hypothetical);
            }
        }

        log.debug("[RAG] 查询改写完成：1 个原始查询 → {} 个检索查询", queries.size());

        // ---------- [2] 多路召回 ----------
        List<List<ScoredDocument>> channels = new ArrayList<>();

        for (String q : queries) {
            // 通道 A：稠密检索（向量）——索引不可用时跳过，由 BM25 单路兜底
            if (denseAvailable) {
                channels.add(denseSearch(q, ragProperties.getDenseTopK()));
            }

            // 通道 B：稀疏检索（BM25）
            channels.add(sparseRetriever.search(q, ragProperties.getSparseTopK()));
        }

        // ---------- [3] RRF 融合 ----------
        List<ScoredDocument> fused = rrfFusion.fuse(channels, ragProperties.getFusionTopK());

        if (fused.isEmpty()) {
            log.debug("[RAG] 未检索到任何文档");
            return List.of();
        }

        // ---------- [4] 重排序 ----------
        List<ScoredDocument> reranked = ragProperties.isEnableRerank()
                ? reranker.rerank(question, fused, ragProperties.getRerankTopN())
                : fused.stream().limit(ragProperties.getRerankTopN()).toList();

        long elapsed = System.currentTimeMillis() - start;
        log.info("[RAG] 检索完成：{} 个查询 → 融合 {} 篇 → 精排 {} 篇，耗时 {}ms",
                queries.size(), fused.size(), reranked.size(), elapsed);

        return reranked;
    }

    /**
     * 稠密检索（向量相似度）。
     *
     * <p>注意 Spring AI 的向量检索返回的 metadata 中，{@code distance}
     * 的语义<b>取决于底层向量库</b>（可能是余弦距离、内积或 L2）。
     * 这里统一按「距离越小越相关」处理，转换为「分数越大越相关」，
     * 保证与 BM25 的分数方向一致——尽管 RRF 只用排名，
     * 但统一方向能让调试输出更直观。</p>
     */
    private List<ScoredDocument> denseSearch(String query, int topK) {
        try {
            List<Document> docs = vectorStore.similaritySearch(
                    SearchRequest.builder()
                            .query(query)
                            .topK(topK)
                            // 阈值过滤：剔除明显无关的结果，减少后续融合的噪声
                            .similarityThreshold(ragProperties.getSimilarityThreshold())
                            .build());

            if (docs == null || docs.isEmpty()) {
                return List.of();
            }

            List<ScoredDocument> result = new ArrayList<>();
            for (int i = 0; i < docs.size(); i++) {
                Document doc = docs.get(i);
                double score = extractSimilarity(doc, i, docs.size());
                result.add(ScoredDocument.of(doc, score, ScoredDocument.Source.DENSE));
            }
            return result;

        } catch (Exception e) {
            // 单通道失败不应中断整条链路——混合检索的鲁棒性优势正在于此
            log.warn("[RAG] 稠密检索失败，该通道降级为空: {}", e.getMessage());
            return List.of();
        }
    }

    /**
     * 从 Document 的 metadata 中提取相似度分数。
     *
     * <p>不同 VectorStore 实现写入的键名与语义不同（distance / score），
     * 因此做兼容处理；取不到时按名次线性衰减给一个保守估计值，
     * 保证排序不会被打乱。</p>
     */
    private double extractSimilarity(Document doc, int rank, int total) {
        Object distance = doc.getMetadata().get("distance");
        if (distance instanceof Number n) {
            // 距离 → 相似度：1 - distance，并夹到 [0,1]
            return Math.max(0, Math.min(1, 1.0 - n.doubleValue()));
        }

        Object score = doc.getMetadata().get("score");
        if (score instanceof Number n) {
            return Math.max(0, Math.min(1, n.doubleValue()));
        }

        // 兜底：按名次线性衰减，仅用于展示
        return total <= 1 ? 1.0 : 1.0 - (double) rank / total;
    }

    // ==================== 对外问答接口 ====================

    /**
     * RAG 问答（强化版，推荐使用）。
     *
     * <p>特点：查询改写 + 混合检索 + 重排序 + 引用溯源。</p>
     *
     * @param question 用户问题
     * @return 基于知识库的回答，末尾附带引用来源
     */
    public String ask(String question) {
        if (allChunks.isEmpty()) {
            return "⚠️ 知识库未初始化，请确保 rag-docs 目录下有文档文件。";
        }

        List<ScoredDocument> docs = retrieve(question);

        if (docs.isEmpty()) {
            return "未在知识库中找到相关信息。请尝试换一种问法，或确认该内容是否已收录。";
        }

        String context = buildContext(docs);

        String prompt = """
                你是知识库问答助手。请严格依据下面的【参考资料】回答用户问题。

                【回答要求】
                1. 只使用参考资料中的信息，不要引入外部知识
                2. 如果参考资料中没有相关信息，直接说明"资料中未涉及"，不要推测
                3. 回答末尾用 [1][2] 的形式标注引用的资料编号
                4. 使用中文，条理清晰

                【参考资料】
                %s

                【用户问题】
                %s
                """.formatted(context, question);

        return chatClient.prompt()
                .user(prompt)
                .call()
                .content();
    }

    /**
     * RAG 问答（基础版，保留用于对比教学）。
     *
     * <p>只用向量检索 + 直接拼接上下文，不涉及改写、融合、重排。
     * 把它与 {@link #ask(String)} 并列调用，可直观感受每个优化环节的贡献。</p>
     */
    public String askBaseline(String question) {
        if (allChunks.isEmpty()) {
            return "⚠️ 知识库未初始化。";
        }

        // 基础版依赖向量检索，此处同样惰性构建
        if (!ensureVectorIndex()) {
            return "⚠️ 基础版（纯向量检索）不可用："
                    + (vectorIndexError == null ? "向量索引未就绪" : vectorIndexError)
                    + "\n提示：可改用 /api/rag/ask（会自动降级为 BM25 检索）。";
        }

        List<Document> docs = vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query(question)
                        .topK(ragProperties.getBaselineTopK())
                        .build());

        if (docs == null || docs.isEmpty()) {
            return "未找到相关知识，请尝试换一种问法。";
        }

        StringBuilder context = new StringBuilder();
        for (int i = 0; i < docs.size(); i++) {
            context.append("【参考资料").append(i + 1).append("】\n")
                    .append(docs.get(i).getText()).append("\n\n");
        }

        String prompt = """
                请根据以下参考资料回答用户问题。
                如果参考资料中没有相关信息，请如实告知，不要编造。

                参考资料：
                %s

                用户问题：%s

                请用中文回答，并在回答末尾标注引用的资料编号。
                """.formatted(context, question);

        return chatClient.prompt()
                .user(prompt)
                .call()
                .content();
    }

    /**
     * 检索调试接口：返回检索过程的完整可观测信息。
     *
     * <p>调优 RAG 时，只看「最终答案」是远远不够的——必须能看到
     * 「每一路召回了什么、融合后排名如何变化、重排又做了什么」。
     * 这正是 RAG 调试中最费时间的部分。</p>
     *
     * @param question 用户问题
     * @return 结构化的检索诊断信息
     */
    public Map<String, Object> explain(String question) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("question", question);

        if (allChunks.isEmpty()) {
            result.put("error", "知识库未初始化");
            return result;
        }

        long start = System.currentTimeMillis();

        boolean denseAvailable = ensureVectorIndex();
        result.put("denseAvailable", denseAvailable);
        if (!denseAvailable) {
            result.put("denseUnavailableReason", vectorIndexError);
        }

        // 分别展示两条通道的原始召回，便于判断"是哪一路救了这个结果"
        List<ScoredDocument> dense = denseAvailable
                ? denseSearch(question, ragProperties.getDenseTopK())
                : List.of();
        List<ScoredDocument> sparse = sparseRetriever.search(question, ragProperties.getSparseTopK());

        result.put("denseHits", toBriefList(dense));
        result.put("sparseHits", toBriefList(sparse));

        List<List<ScoredDocument>> channels = new ArrayList<>();
        channels.add(dense);
        channels.add(sparse);
        List<ScoredDocument> fused = rrfFusion.fuse(channels, ragProperties.getFusionTopK());
        result.put("fusedHits", toBriefList(fused));

        List<ScoredDocument> reranked = ragProperties.isEnableRerank()
                ? reranker.rerank(question, fused, ragProperties.getRerankTopN())
                : fused.stream().limit(ragProperties.getRerankTopN()).toList();
        result.put("rerankedHits", toBriefList(reranked));

        result.put("elapsedMs", System.currentTimeMillis() - start);
        result.put("knowledgeBaseSize", allChunks.size());

        return result;
    }

    /**
     * 构建送入 LLM 的上下文块。
     *
     * <p>每个分片前标注来源文件名与编号，使模型能够按要求
     * 输出 {@code [1][2]} 形式的引用，最终实现答案可溯源。</p>
     */
    private String buildContext(List<ScoredDocument> docs) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < docs.size(); i++) {
            Document doc = docs.get(i).document();
            Object source = doc.getMetadata().getOrDefault("source", "未知来源");

            sb.append("【参考资料").append(i + 1)
                    .append("｜来源: ").append(source).append("】\n")
                    .append(doc.getText())
                    .append("\n\n");
        }
        return sb.toString();
    }

    /** 把检索结果转为可 JSON 序列化的简要结构，避免暴露完整 Document 对象 */
    private List<Map<String, Object>> toBriefList(List<ScoredDocument> docs) {
        List<Map<String, Object>> list = new ArrayList<>();
        for (int i = 0; i < docs.size(); i++) {
            ScoredDocument sd = docs.get(i);
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("rank", i + 1);
            item.put("id", sd.id());
            item.put("score", Math.round(sd.score() * 10000) / 10000.0);
            item.put("source", sd.document().getMetadata().getOrDefault("source", ""));
            item.put("preview", truncate(sd.text(), 120));
            list.add(item);
        }
        return list;
    }

    private String truncate(String s, int max) {
        if (s == null) {
            return "";
        }
        String flat = s.replaceAll("\\s+", " ").trim();
        return flat.length() <= max ? flat : flat.substring(0, max) + "...";
    }

    // ==================== 供评估模块调用 ====================

    /** 返回知识库全部分片，评估模块用于构造「标准答案应召回哪些分片」 */
    public List<Document> getAllChunks() {
        return allChunks;
    }

    /** 知识库是否就绪（BM25 索引可用即视为就绪，向量索引可选） */
    public boolean isReady() {
        return !allChunks.isEmpty();
    }

    /** 向量索引（稠密检索）是否可用 */
    public boolean isVectorIndexReady() {
        return vectorIndexReady;
    }

    /** 向量索引构建失败原因，null 表示无错误 */
    public String getVectorIndexError() {
        return vectorIndexError;
    }

    /** 降级用的 embedding 检索：仅供调试 */
    public EmbeddingModel embeddingModel() {
        return embeddingModel;
    }
}
