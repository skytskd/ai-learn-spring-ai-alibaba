package com.ailearn.alibaba.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * <h1>RagProperties — RAG 流水线可调参数</h1>
 *
 * <p>把 RAG 调优中所有需要反复实验的参数集中到一处，并支持通过
 * {@code application.yml} 覆盖。这是 RAG 调优的基本功——
 * <b>参数必须可外部化配置，否则每次调参都要改代码、重编译。</b></p>
 *
 * <h2>调参优先级建议</h2>
 * <p>按投入产出比排序，实践中应按此顺序逐个调整：</p>
 * <ol>
 *   <li>{@code chunkSize} + 切分策略 —— 影响最大，且不可后期弥补</li>
 *   <li>{@code enableRerank} —— 打开即有提升，收益最直接</li>
 *   <li>{@code rerankTopN} —— 直接影响送入 LLM 的上下文质量</li>
 *   <li>{@code enableQueryExpansion} / {@code enableHyde} —— 提升召回覆盖</li>
 *   <li>{@code denseTopK} / {@code sparseTopK} —— 在融合前调整各通道候选量</li>
 * </ol>
 *
 * <p>⚠️ 每次只改一个参数，并用评估模块量化效果。同时改多个参数
 * 会导致无法归因——这是调参最常见的错误。</p>
 *
 * @author ai-learn
 */
@Data
@Component
@ConfigurationProperties(prefix = "ai-learn.rag")
public class RagProperties {

    // ==================== 文档切分 ====================

    /**
     * 每个分片的目标 token 数。
     *
     * <p>经验区间 200 ~ 800：</p>
     * <ul>
     *   <li>太小（&lt; 200）：单块语义不完整，模型缺乏上下文</li>
     *   <li>太大（&gt; 1000）：单块混入多主题，向量表示被"平均"稀释，
     *       检索精度显著下降——这是最常见也最致命的错误</li>
     * </ul>
     *
     * <p>中文场景下 1 个汉字通常占 1~2 个 token，因此 500 token
     * 约等于 300~500 个汉字。</p>
     */
    private int chunkSize = 500;

    /** 分片的最小字符数，低于此值不单独成块，会与相邻块合并 */
    private int minChunkSizeChars = 100;

    /** 小于该长度（token 数）的文本不参与向量化，避免产生无意义的噪声块 */
    private int minChunkLengthToEmbed = 5;

    /** 单篇文档最多切分成多少块，防止超大文件产生过多分片 */
    private int maxNumChunks = 10000;

    // ==================== 各通道候选数 ====================

    /**
     * 稠密检索（向量）每路查询返回的候选数。
     * 设得比最终需要的大，为后续融合和重排留足空间。
     */
    private int denseTopK = 10;

    /**
     * 稀疏检索（BM25）每路查询返回的候选数。
     * 通常略小于稠密通道，因为 BM25 的头部精度较高，
     * 长尾部分噪声增长更快。
     */
    private int sparseTopK = 10;

    /**
     * RRF 融合后保留的候选数。
     * 这是「送入重排器的候选规模」——太少则重排无从选择，
     * 太多则重排成本（延迟 / token）急剧上升。
     */
    private int fusionTopK = 20;

    /**
     * 重排序后最终送入 LLM 的文档数。
     *
     * <p>这是 RAG 中<b>最需要精调</b>的参数之一：</p>
     * <ul>
     *   <li>太少（1~2）：信息不足，模型无法完整作答</li>
     *   <li>太多（&gt; 8）：无关内容稀释注意力，幻觉率上升，
     *       且 token 成本线性增长</li>
     *   <li>经验值 3 ~ 5</li>
     * </ul>
     */
    private int rerankTopN = 5;

    /** 基础版（对照用）RAG 的检索条数 */
    private int baselineTopK = 4;

    /** 向量检索的相似度阈值，低于该值的候选直接丢弃（0 表示不过滤） */
    private double similarityThreshold = 0.0;

    // ==================== 功能开关 ====================

    /** 是否启用重排序。关闭后融合结果直接截断，可用于 A/B 对比验证重排收益 */
    private boolean enableRerank = true;

    /** 是否启用多查询扩展 */
    private boolean enableQueryExpansion = false;

    /**
     * 多查询扩展生成的变体数量（不含原查询）。
     *
     * <p>每个变体都会触发一轮双通道检索，因此检索成本
     * ≈ (1 + count) × 2 次查询。建议 2 ~ 3。</p>
     */
    private int queryExpansionCount = 2;

    /**
     * 是否启用 HyDE。
     *
     * <p>注意：HyDE 需要额外一次 LLM 调用（生成假想文档），
     * 会明显增加首字延迟。对延迟敏感的场景建议关闭。</p>
     */
    private boolean enableHyde = false;
}
