package com.ailearn.alibaba.rag;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * <h1>Bm25SparseRetriever — 稀疏检索（BM25）</h1>
 *
 * <p>纯向量检索最大的短板是<b>专有名词、型号、编号、代码符号</b>的匹配。
 * 例如查询 "text-embedding-v2 的计费" 或 "ERR_403 怎么解决"，
 * 向量模型很可能把语义相近但实际无关的段落排到前面，
 * 因为它把 "text-embedding-v2" 这种标识符压成了一个模糊的语义点。</p>
 *
 * <p>BM25（Best Matching 25）是经典概率检索模型，基于词频统计，
 * 对这类「精确 token」的匹配极其敏感。两者互补构成<b>混合检索</b>。</p>
 *
 * <h2>BM25 公式</h2>
 * <pre>
 *                      f(t,d) · (k1 + 1)
 *   score(t,d) = IDF(t) · ────────────────────────
 *                        f(t,d) + k1 · (1 - b + b · |d|/avgdl)
 *
 *   其中：
 *     f(t,d)  = 词 t 在文档 d 中的出现次数（词频 TF）
 *     |d|     = 文档 d 的长度（词数）
 *     avgdl   = 所有文档的平均长度
 *     k1      = TF 饱和参数，控制词频增长的收益递减（典型 1.2 ~ 2.0）
 *     b       = 长度归一化强度，0 表示不归一化，1 表示完全归一化（典型 0.75）
 *     IDF(t)  = 逆文档频率，衡量词 t 的区分度
 * </pre>
 *
 * <h2>IDF 的实现选择</h2>
 * <p>标准 BM25 的 IDF 为 {@code ln((N - n + 0.5) / (n + 0.5) + 1)}，
 * 其中 N 为文档总数、n 为包含词 t 的文档数。这里额外加 1 是为了
 * 保证 IDF 恒为正——否则当某词出现在超过一半文档中时 IDF 会变负，
 * 导致「包含该词的文档反而扣分」的反直觉行为。</p>
 *
 * <h2>中文分词说明</h2>
 * <p>中文没有空格，标准做法需接入分词器（HanLP / IK / jieba）。
 * 本项目为教学目的，采用<b>字符二元组（bigram）+ 英文数字 token</b> 的
 * 轻量方案，可覆盖绝大多数中文检索场景，且零外部依赖。
 * 生产环境建议替换 {@link #tokenize(String)} 为真实分词器。</p>
 *
 * @author ai-learn
 */
@Slf4j
@Component
public class Bm25SparseRetriever {

    /** TF 饱和参数：越大则词频增长带来的增益越持久 */
    private final double k1 = 1.5;

    /** 长度归一化系数：0.75 为文献中的经验最优值 */
    private final double b = 0.75;

    /** 倒排索引：token → 文档 ID 集合（用于快速计算文档频率 DF） */
    private final Map<String, Set<String>> invertedIndex = new HashMap<>();

    /** 文档 ID → 词频表 */
    private final Map<String, Map<String, Integer>> docTermFreq = new HashMap<>();

    /** 文档 ID → 文档长度（token 数） */
    private final Map<String, Integer> docLength = new HashMap<>();

    /** 文档 ID → 原始 Document（用于返回结果） */
    private final Map<String, Document> docStore = new LinkedHashMap<>();

    /** 全部文档的平均长度，在 {@link #build} 中计算 */
    private double avgDocLength = 0;

    /**
     * 构建索引。应在知识库加载完成后调用一次。
     *
     * <p>时间复杂度 O(N · L)，N 为文档数、L 为平均文档长度。
     * 对万级分片规模，构建耗时在百毫秒量级，可接受。</p>
     *
     * @param documents 全部文档分片
     */
    public synchronized void build(List<Document> documents) {
        // 重建前先清空，支持知识库热更新
        invertedIndex.clear();
        docTermFreq.clear();
        docLength.clear();
        docStore.clear();

        for (Document doc : documents) {
            String id = doc.getId();
            docStore.put(id, doc);

            Map<String, Integer> tf = new HashMap<>();
            List<String> tokens = tokenize(doc.getText());
            for (String token : tokens) {
                tf.merge(token, 1, Integer::sum);
                invertedIndex.computeIfAbsent(token, k -> new HashSet<>()).add(id);
            }
            docTermFreq.put(id, tf);
            docLength.put(id, tokens.size());
        }

        avgDocLength = docLength.values().stream()
                .mapToInt(Integer::intValue)
                .average()
                .orElse(0);

        log.info("[BM25] 索引构建完成：{} 篇文档，{} 个唯一 token，平均长度 {}",
                documents.size(), invertedIndex.size(), String.format("%.1f", avgDocLength));
    }

    /**
     * 稀疏检索主入口。
     *
     * @param query 查询语句
     * @param topK  返回条数
     * @return 按 BM25 分数降序排列的结果；索引为空时返回空列表（调用方需容错）
     */
    public List<ScoredDocument> search(String query, int topK) {
        if (docStore.isEmpty() || query == null || query.isBlank()) {
            return List.of();
        }

        List<String> queryTokens = tokenize(query);
        int totalDocs = docStore.size();

        // 文档 ID → 累计 BM25 分数
        Map<String, Double> scores = new HashMap<>();

        for (String token : queryTokens) {
            Set<String> containing = invertedIndex.get(token);
            if (containing == null || containing.isEmpty()) {
                // 该 token 在任何文档中都不存在 —— 对检索无贡献，跳过
                continue;
            }

            double idf = idf(totalDocs, containing.size());

            for (String docId : containing) {
                int tf = docTermFreq.get(docId).getOrDefault(token, 0);
                int len = docLength.get(docId);
                double norm = 1 - b + b * (len / avgDocLength);
                double termScore = idf * (tf * (k1 + 1)) / (tf + k1 * norm);
                scores.merge(docId, termScore, Double::sum);
            }
        }

        if (scores.isEmpty()) {
            log.debug("[BM25] 查询「{}」未命中任何 token", query);
            return List.of();
        }

        // 归一化到 [0,1]，便于与稠密分数在同一量纲下观察
        double maxScore = scores.values().stream().mapToDouble(Double::doubleValue).max().orElse(1.0);

        return scores.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(topK)
                .map(e -> ScoredDocument.of(
                        docStore.get(e.getKey()),
                        maxScore > 0 ? e.getValue() / maxScore : 0,
                        ScoredDocument.Source.SPARSE))
                .toList();
    }

    /**
     * IDF 计算，采用加 1 平滑保证恒正。
     *
     * @param totalDocs      文档总数 N
     * @param docsWithToken  包含该 token 的文档数 n
     */
    private double idf(int totalDocs, int docsWithToken) {
        return Math.log((totalDocs - docsWithToken + 0.5) / (docsWithToken + 0.5) + 1.0);
    }

    /**
     * 轻量分词：英文/数字按标识符整体切分，中文切字符二元组（bigram）。
     *
     * <p>示例：{@code "text-embedding-v2 是阿里云的模型"}
     * → {@code [text-embedding-v2, 是阿, 阿里, 里云, 云的, 的模, 模型]}</p>
     *
     * <p>之所以用 bigram 而非 unigram，是因为 unigram 会把「阿里」和「里云」
     * 拆成无区分度的单字，导致几乎所有中文文档都命中同一个字而 IDF 趋近 0。
     * bigram 保留了局部语序信息，检索质量显著更好。</p>
     */
    private List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        if (text == null) {
            return tokens;
        }

        String lower = text.toLowerCase();

        // 1) 抽取英文单词 / 数字 / 连字符标识符（如 text-embedding-v2、ER_403、gpt-4o）
        java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("[a-z0-9][a-z0-9_.-]*")
                .matcher(lower);
        while (m.find()) {
            tokens.add(m.group());
        }

        // 2) 抽取中文连续片段，切成 bigram
        java.util.regex.Matcher cn = java.util.regex.Pattern
                .compile("[\\u4e00-\\u9fa5]+")
                .matcher(lower);
        while (cn.find()) {
            String seg = cn.group();
            if (seg.length() == 1) {
                tokens.add(seg);
            } else {
                for (int i = 0; i + 1 < seg.length(); i++) {
                    tokens.add(seg.substring(i, i + 2));
                }
            }
        }

        return tokens;
    }

    /** 当前索引的文档数量，供评估与健康检查使用 */
    public int size() {
        return docStore.size();
    }

    /** 排序辅助：按分数降序 */
    public static Comparator<ScoredDocument> byScoreDesc() {
        return Comparator.comparingDouble(ScoredDocument::score).reversed();
    }
}
