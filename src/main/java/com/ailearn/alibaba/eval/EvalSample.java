package com.ailearn.alibaba.eval;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * <h1>EvalSample — 评估样本（金标准）</h1>
 *
 * <p>评估体系的地基。没有它，一切「RAG 效果变好了」的说法都是主观臆断。</p>
 *
 * <h2>为什么必须要金标准？</h2>
 * <p>RAG 调优的常见失败模式是：改了参数，试了两个问题，
 * 感觉「好像好了一点」，就推定改进有效。这是典型的<b>过拟合直觉</b>——
 * 在积累到一定规模前，主观感受无法区分真实提升与随机波动。</p>
 *
 * <p>有金标准后，每次改动都能得到一个数值，且能进入 CI 做回归防护。</p>
 *
 * <h2>样本构成</h2>
 * <pre>
 *   EvalSample
 *     ├─ question          用户问题
 *     ├─ groundTruthAnswer 标准答案（人工撰写，用于评估生成质量）
 *     └─ relevantChunkKeys 应该被召回的分片标识（用于评估检索质量）
 * </pre>
 *
 * <h2>relevantChunkKeys 怎么定？</h2>
 * <p>两种做法：</p>
 * <ol>
 *   <li><b>人工标注</b>（推荐）：人工阅读知识库，标出每个问题
 *       真正相关的分片。准确但费时，适合小规模高价值数据集。</li>
 *   <li><b>关键短语匹配</b>（自动）：为每个问题指定若干个
 *       "必须在正确答案中出现的短语"，检索时判断命中的分片是否包含这些短语。
 *       本项目采用这种方式，因为它在文档变动时更鲁棒——
 *       分片 ID 会因切分参数变化而改变，但关键短语不会。</li>
 * </ol>
 *
 * <p>本类同时支持两种：{@code keyPhrases} 用于自动判定，
 * {@code relevantChunkKeys} 用于人工精确标注。</p>
 *
 * @author ai-learn
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class EvalSample {

    /** 样本唯一标识，便于在报告中定位问题样本 */
    private String id;

    /** 评估问题 */
    private String question;

    /**
     * 标准答案。
     *
     * <p>用于 Answer Relevancy 与 Faithfulness 的评估——
     * 注意这两个指标的参照不同：</p>
     * <ul>
     *   <li>Answer Relevancy 参照<b>问题</b></li>
     *   <li>Faithfulness 参照<b>检索到的上下文</b></li>
     * </ul>
     * <p>标准答案在这里主要用于人工核对与 Context Recall 的辅助判断。</p>
     */
    private String groundTruthAnswer;

    /**
     * 关键短语列表：正确答案中必然包含的表述。
     *
     * <p>用于自动判定「检索是否命中正确内容」——
     * 只要召回的分片包含其中任一短语，就认为该分片是相关的。</p>
     */
    private List<String> keyPhrases;

    /**
     * 人工标注的相关分片 ID（可选，精确评估时使用）。
     */
    private List<String> relevantChunkKeys;

    /** 样本分类标签，便于分组分析（如 "计费"、"模型"、"配置"） */
    private String category;

    /**
     * 构造简单样本的便捷方法。
     */
    public static EvalSample of(String id, String question, String answer, String category, String... keyPhrases) {
        EvalSample s = new EvalSample();
        s.setId(id);
        s.setQuestion(question);
        s.setGroundTruthAnswer(answer);
        s.setCategory(category);
        s.setKeyPhrases(List.of(keyPhrases));
        s.setRelevantChunkKeys(List.of());
        return s;
    }

    /**
     * 判断一个文本片段是否与本题相关（基于关键短语）。
     *
     * @param text 待判定的文本（通常是检索到的分片内容）
     * @return 命中任一关键短语即视为相关
     */
    public boolean matches(String text) {
        if (text == null || keyPhrases == null || keyPhrases.isEmpty()) {
            return false;
        }
        String lower = text.toLowerCase();
        return keyPhrases.stream()
                .anyMatch(phrase -> lower.contains(phrase.toLowerCase()));
    }
}
