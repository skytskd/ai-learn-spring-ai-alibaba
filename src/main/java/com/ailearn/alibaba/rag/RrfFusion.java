package com.ailearn.alibaba.rag;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * <h1>RrfFusion — 倒数排名融合（Reciprocal Rank Fusion）</h1>
 *
 * <p>把多路召回的排序结果合并成一份排序。这是混合检索的关键环节。</p>
 *
 * <h2>为什么不直接把分数加起来？</h2>
 * <p>稠密检索的分数（余弦相似度，约 0.6 ~ 0.9）与稀疏检索的 BM25 分数
 * （归一化前后量纲完全不同）<b>不可直接相加</b>。强行加权求和需要
 * 人工调参，且换个数据集就要重调，鲁棒性差。</p>
 *
 * <h2>RRF 的核心思想</h2>
 * <p>RRF 完全<b>抛弃分数数值，只用排名</b>。这在信息检索领域称为
 * 「分数不可比问题」的经典解法：</p>
 *
 * <pre>
 *                   1
 *   RRF(d) = Σ  ──────────
 *            r∈R   k + rank_r(d)
 *
 *   R = 所有召回通道的集合（稠密、稀疏……）
 *   rank_r(d) = 文档 d 在通道 r 中的排名（从 1 开始）
 *   k = 平滑常数，经典取 60
 * </pre>
 *
 * <h2>k=60 的作用</h2>
 * <p>k 决定了「排名差异的衰减速度」：</p>
 * <ul>
 *   <li>k 很小（如 1）→ 第 1 名权重 1/2，第 10 名 1/11，差距被极度放大，
 *       融合结果几乎等于「各通道第 1 名的内斗」</li>
 *   <li>k 很大（如 1000）→ 各名次得分趋近，退化为「计数投票」</li>
 *   <li>k = 60 → 经验最优平衡点，来自 Cormack 等人 2009 年的原始论文</li>
 * </ul>
 *
 * <h2>RRF 的三大优势</h2>
 * <ol>
 *   <li><b>免调参</b>：无需为不同检索器归一化分数，开箱即用</li>
 *   <li><b>抗异常</b>：某通道给出离谱的高分不会污染融合结果，只看排名</li>
 *   <li><b>奖励共识</b>：同时被稠密和稀疏两路召回、且排名都靠前的文档会获得
 *       双重加分——这正是「语义相关 + 关键词命中」的最优样本</li>
 * </ol>
 *
 * @author ai-learn
 */
@Slf4j
@Component
public class RrfFusion {

    /**
     * 平滑常数 k。
     *
     * <p>取 60 来自 RRF 原始论文（Cormack et al., SIGIR 2009）的经验值，
     * 在 TREC 系列评测中被反复验证。除非有明确的数据集调优依据，否则不建议修改。</p>
     */
    private static final int K = 60;

    /**
     * 融合多路召回结果。
     *
     * @param channels 多个召回通道的结果，每个内层 List 已按相关性降序排列
     * @param topK     融合后保留的条数（通常设得比最终需要的多，留给重排阶段筛选）
     * @return 按 RRF 分数降序排列的融合结果
     */
    public List<ScoredDocument> fuse(List<List<ScoredDocument>> channels, int topK) {
        // 文档 ID → RRF 累计分数
        Map<String, Double> rrfScores = new LinkedHashMap<>();
        // 文档 ID → 原始 Document（避免多路返回同一文档时重复存储）
        Map<String, ScoredDocument> registry = new LinkedHashMap<>();

        for (List<ScoredDocument> channel : channels) {
            if (channel == null || channel.isEmpty()) {
                continue;
            }

            for (int rank = 0; rank < channel.size(); rank++) {
                ScoredDocument sd = channel.get(rank);
                String id = sd.id();

                // rank 从 0 开始，公式中从 1 开始，故 +1
                double contribution = 1.0 / (K + rank + 1);
                rrfScores.merge(id, contribution, Double::sum);

                // 同一个文档可能被多路召回，保留首次出现的实例即可
                registry.putIfAbsent(id, sd);
            }
        }

        if (rrfScores.isEmpty()) {
            log.debug("[RRF] 所有召回通道均为空，融合结果为空");
            return List.of();
        }

        List<ScoredDocument> fused = new ArrayList<>();
        for (Map.Entry<String, Double> e : rrfScores.entrySet()) {
            fused.add(ScoredDocument.of(
                    registry.get(e.getKey()).document(),
                    e.getValue(),
                    ScoredDocument.Source.FUSION));
        }

        fused.sort(Bm25SparseRetriever.byScoreDesc());

        // 归一化显示分数到 [0,1]。注意：归一化仅影响可读性，
        // 不改变排序，因为排序在归一化之前已完成。
        double max = fused.get(0).score();
        List<ScoredDocument> result = fused.stream()
                .limit(topK)
                .map(sd -> ScoredDocument.of(sd.document(), max > 0 ? sd.score() / max : 0,
                        ScoredDocument.Source.FUSION))
                .toList();

        log.debug("[RRF] 融合 {} 个通道，去重后 {} 篇，返回 Top-{}",
                channels.size(), rrfScores.size(), result.size());

        return result;
    }
}
