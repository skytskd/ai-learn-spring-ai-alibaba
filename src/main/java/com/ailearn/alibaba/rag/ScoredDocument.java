package com.ailearn.alibaba.rag;

import org.springframework.ai.document.Document;

/**
 * <h1>ScoredDocument — 带分数的检索结果</h1>
 *
 * <p>RAG 调优的基石：所有检索通道（稠密 / 稀疏 / 重排）统一返回本类型，
 * 便于融合、排序、对比与指标计算。</p>
 *
 * <h2>为什么需要它？</h2>
 * <p>Spring AI 的 {@code VectorStore.similaritySearch()} 只返回 {@code List<Document>}，
 * 分数虽存放在 metadata 的 {@code distance} 字段里，但语义模糊（不同向量库
 * 含义不一致：可能是余弦距离、可能是内积、可能是 L2）。</p>
 *
 * <p>我们在检索层就把分数显式抽出来，并统一约定：<b>分数越大越相关</b>。
 * 稠密检索返回的是「距离」时，这里会做一次 {@code 1 - distance} 的归一转换。</p>
 *
 * @param document 命中的文档分片
 * @param score    相关性分数，越大越相关，范围 [0, 1]（重排后同样归一化）
 * @param source   该结果的来源通道，用于排查「是哪一路召回救了这个结果」
 *
 * @author ai-learn
 */
public record ScoredDocument(Document document, double score, Source source) {

    /** 检索通道来源 */
    public enum Source {
        /** 稠密检索：向量相似度（语义匹配强，专有名词弱） */
        DENSE,
        /** 稀疏检索：BM25 词频统计（专有名词、编号、代码符号强） */
        SPARSE,
        /** 融合结果：多路召回经 RRF 合并 */
        FUSION,
        /** 重排结果：经交叉编码器 / LLM 精排 */
        RERANK
    }

    public static ScoredDocument of(Document doc, double score, Source source) {
        return new ScoredDocument(doc, score, source);
    }

    public String text() {
        return document.getText();
    }

    public String id() {
        return document.getId();
    }

    /**
     * 返回一份替换了来源标记的副本，用于在流水线各阶段之间传递。
     */
    public ScoredDocument withSource(Source newSource) {
        return new ScoredDocument(document, score, newSource);
    }
}
