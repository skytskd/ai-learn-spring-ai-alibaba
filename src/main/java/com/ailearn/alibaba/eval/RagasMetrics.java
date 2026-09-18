package com.ailearn.alibaba.eval;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * <h1>RagasMetrics — RAGAS 四指标的 LLM-as-Judge 实现</h1>
 *
 * <p>RAGAS（Retrieval Augmented Generation Assessment，arXiv 2309.15217）
 * 是 RAG 评估领域事实上的标准框架。它用「无参考指标」的思路
 * 绕开了人工标注标准答案的成本问题。</p>
 *
 * <h2>四个核心指标</h2>
 *
 * <h3>1. Faithfulness（忠实度）</h3>
 * <pre>
 *   Faithfulness = 回答中能从上下文得到支撑的论断数 / 回答中的论断总数
 * </pre>
 * <p><b>这是幻觉的直接度量</b>，也是最该优先关注的指标。它的分母是
 * 「回答里的论断总数」，所以一个有 5 句话但有 4 句是编造的回答，
 * 忠实度只有 0.2。</p>
 *
 * <p>注意 Faithfulness 与「答案是否正确」<b>无关</b>——如果上下文本身是错的，
 * 忠实于错误上下文的回答忠实度仍是 1.0。忠实度衡量的是
 * 「有没有超出给定材料」，不是「说的对不对」。</p>
 *
 * <h3>2. Answer Relevancy（答案相关性）</h3>
 * <pre>
 *   Answer Relevancy = 回答与问题的契合程度
 * </pre>
 * <p>衡量回答是否「切题」。它的典型反面是：回答很流畅、很充实、
 * 也没编造，但答非所问——比如问「怎么收费」却回答了一堆
 * 「平台介绍」的内容。</p>
 *
 * <p>RAGAS 原论文的做法是让 LLM 基于回答<b>反推问题</b>，
 * 再计算反推问题与原问题的相似度。本实现直接让 LLM 打分，
 * 更省调用但精度略低。</p>
 *
 * <h3>3. Context Precision（上下文精确率）</h3>
 * <pre>
 *   Context Precision = 上下文中真正有用的分片排在前面的程度
 * </pre>
 * <p>衡量<b>排序质量</b>：有用的分片是否被排在了前面。如果相关分片
 * 排在 5 个无关分片之后，即便召回率满分，精确率也很低——
 * 而 LLM 的注意力对位置敏感，排后面的内容容易被忽略。</p>
 *
 * <h3>4. Context Recall（上下文召回率）</h3>
 * <pre>
 *   Context Recall = 标准答案中的信息有多少能从上下文找到
 * </pre>
 * <p>衡量<b>上下文覆盖度</b>。这是唯一需要标准答案的指标。
 * 如果召回率低，说明检索环节漏了内容，模型无论多强都答不全。</p>
 *
 * <h2>LLM-as-Judge 的已知偏差（必须知道）</h2>
 * <p>参考：Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and
 * Chatbot Arena"（arXiv 2306.05685）</p>
 * <ol>
 *   <li><b>位置偏差</b>：倾向于偏好排在前面的选项
 *       → 缓解：随机化顺序，或对调顺序求平均</li>
 *   <li><b>冗长偏差</b>：倾向于给长回答更高分
 *       → 缓解：在 prompt 中明确"长度不影响评分"</li>
 *   <li><b>自我增强偏差</b>：偏好与自己风格相似的输出
 *       → 缓解：用不同模型做裁判</li>
 *   <li><b>分数聚集</b>：总是给 7-8 分，区分度低
 *       → 缓解：用离散档位（0/0.5/1）而非连续分数</li>
 * </ol>
 * <p>因此 LLM-as-Judge 的结果应<b>看趋势而非绝对值</b>。
 * 它适合回答「这次改动是变好还是变差」，不适合回答「当前水平是 0.72」。</p>
 *
 * @author ai-learn
 */
@Slf4j
@Component
public class RagasMetrics {

    private final ChatClient chatClient;

    /** 从 LLM 输出中抓取 0~1 的小数或分数形式（如 "0.8"、"8/10"、"8"） */
    private static final Pattern SCORE_PATTERN = Pattern.compile("(\\d+(?:\\.\\d+)?)\\s*(?:/\\s*10)?");

    public RagasMetrics(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    /**
     * Faithfulness（忠实度）—— 幻觉度量。
     *
     * @param answer  模型生成的回答
     * @param contexts 检索到的上下文分片
     * @return 忠实度，[0, 1]
     */
    public double faithfulness(String answer, List<String> contexts) {
        if (isBlank(answer) || contexts == null || contexts.isEmpty()) {
            return 0.0;
        }

        String contextText = String.join("\n---\n", contexts);

        String prompt = """
                你是一个严谨的事实核查员。请判断【回答】中的每一条陈述，
                是否都能在【参考资料】中找到依据。

                【参考资料】
                %s

                【回答】
                %s

                【判定步骤】
                1. 把回答拆解为若干条独立的陈述
                2. 逐条判断该陈述是否被参考资料支持
                3. 统计：被支持的陈述数 / 陈述总数

                【重要】
                - 只依据参考资料判断，不要使用你自己的知识
                - 参考资料中没有提到的具体数字、名称、结论，一律视为"无依据"
                - 回答中"资料未提及"这类表述算作有依据
                - 请只输出一个小数（0 到 1 之间，保留两位），不要任何解释

                评分：""".formatted(contextText, answer);

        return score(prompt, "Faithfulness");
    }

    /**
     * Answer Relevancy（答案相关性）。
     *
     * @param question 用户问题
     * @param answer   模型回答
     * @return 相关性，[0, 1]
     */
    public double answerRelevancy(String question, String answer) {
        if (isBlank(question) || isBlank(answer)) {
            return 0.0;
        }

        String prompt = """
                你是一个评估专家。请判断【回答】是否直接、切题地回应了【问题】。

                【问题】
                %s

                【回答】
                %s

                【评分标准】
                - 1.0：直接完整地回答了问题核心，没有无关内容
                - 0.8：回答了主要问题，但有少量冗余或偏题
                - 0.5：部分相关，回避了问题核心或答非所问
                - 0.2：基本不相关，只是沾边
                - 0.0：完全无关，或回答是"我不知道"这类无信息内容

                【重要】
                - 回答的长度不影响评分，短而准确的回答应得高分
                - 重点看"是否切题"，不要评价回答的对错
                - 请只输出一个小数（0 到 1 之间，保留两位），不要任何解释

                评分：""".formatted(question, answer);

        return score(prompt, "AnswerRelevancy");
    }

    /**
     * Context Precision（上下文精确率）—— 排序质量。
     *
     * @param question 用户问题
     * @param contexts 按检索顺序排列的上下文（顺序有意义）
     * @return 精确率，[0, 1]
     */
    public double contextPrecision(String question, List<String> contexts) {
        if (isBlank(question) || contexts == null || contexts.isEmpty()) {
            return 0.0;
        }

        StringBuilder numbered = new StringBuilder();
        for (int i = 0; i < contexts.size(); i++) {
            numbered.append("[").append(i + 1).append("] ")
                    .append(truncate(contexts.get(i), 600))
                    .append("\n\n");
        }

        String prompt = """
                你是一个检索质量评估专家。请判断下面这些【资料片段】
                对回答【问题】是否有用，并考虑它们的排列顺序。

                【问题】
                %s

                【资料片段（按检索返回顺序排列）】
                %s

                【评分标准 — 注意顺序权重】
                - 1.0：所有有用片段都排在最前面，无用片段都在后面
                - 0.8：大部分有用片段靠前，个别排序颠倒
                - 0.5：有用与无用片段混杂，顺序上没有明显规律
                - 0.2：有用片段被大量无用片段压在后面
                - 0.0：所有片段都与问题无关

                【重要】
                - 顺序非常重要：有用片段排得越靠前，分数越高
                - 请只输出一个小数（0 到 1 之间，保留两位），不要任何解释

                评分：""".formatted(question, numbered);

        return score(prompt, "ContextPrecision");
    }

    /**
     * Context Recall（上下文召回率）—— 覆盖度。
     *
     * @param groundTruth 标准答案
     * @param contexts    检索到的上下文
     * @return 召回率，[0, 1]
     */
    public double contextRecall(String groundTruth, List<String> contexts) {
        if (isBlank(groundTruth) || contexts == null || contexts.isEmpty()) {
            return 0.0;
        }

        String contextText = String.join("\n---\n", contexts);

        String prompt = """
                你是一个检索覆盖度评估专家。
                请判断【标准答案】中的信息有多少能从【参考资料】中找到。

                【标准答案】
                %s

                【参考资料】
                %s

                【判定步骤】
                1. 把标准答案拆解为若干条关键信息点
                2. 逐条检查该信息点是否出现在参考资料中
                3. 统计：能找到的信息点数 / 信息点总数

                【重要】
                - 只看信息点是否"存在"，不要求表述完全一致（同义表述算命中）
                - 请只输出一个小数（0 到 1 之间，保留两位），不要任何解释

                评分：""".formatted(groundTruth, contextText);

        return score(prompt, "ContextRecall");
    }

    /**
     * Faithfulness 的增强版：对同一输入采样多次取平均。
     *
     * <p>LLM 评分有随机性，单次结果可能波动 ±0.15。对关键评估
     * （如发布前的回归验证），建议采样 3 次取中位数，
     * 可显著降低方差。</p>
     *
     * <p>代价是 3 倍调用成本，因此只用于关键指标。</p>
     */
    public double faithfulnessWithSampling(String answer, List<String> contexts, int samples) {
        if (samples <= 1) {
            return faithfulness(answer, contexts);
        }

        List<Double> scores = new java.util.ArrayList<>();
        for (int i = 0; i < samples; i++) {
            scores.add(faithfulness(answer, contexts));
        }
        scores.sort(Double::compareTo);

        // 取中位数：比平均值更抗离群值（某次模型抽风给了 0 分）
        double median = scores.size() % 2 == 1
                ? scores.get(scores.size() / 2)
                : (scores.get(scores.size() / 2 - 1) + scores.get(scores.size() / 2)) / 2.0;

        log.debug("[RAGAS] Faithfulness 采样 {} 次：{}，中位数 {}",
                samples, scores, String.format("%.3f", median));
        return median;
    }

    // ==================== 内部工具 ====================

    /**
     * 调用 LLM 打分并解析。
     *
     * <p>解析失败时返回 0 而非抛异常——评估过程中单个样本失败
     * 不应中断整轮评估。</p>
     */
    private double score(String prompt, String metricName) {
        try {
            String raw = chatClient.prompt()
                    .user(prompt)
                    .call()
                    .content();
            return parseScore(raw, metricName);

        } catch (Exception e) {
            log.warn("[RAGAS] {} 打分失败: {}", metricName, e.getMessage());
            return 0.0;
        }
    }

    /**
     * 解析 LLM 返回的分数。
     *
     * <p>需要处理多种格式：{@code 0.8}、{@code .8}、{@code 8/10}、
     * {@code 8}、{@code 评分：0.8}。策略是扫描第一个数字，
     * 若大于 1 则假定是 10 分制并做归一化。</p>
     */
    private double parseScore(String raw, String metricName) {
        if (isBlank(raw)) {
            log.warn("[RAGAS] {} 返回空内容", metricName);
            return 0.0;
        }

        String text = raw.trim();

        // 处理 ".8" 这种省略前导零的写法
        text = text.replaceAll("(?<![0-9])\\.(\\d)", "0.$1");

        Matcher m = SCORE_PATTERN.matcher(text);
        if (!m.find()) {
            log.warn("[RAGAS] {} 无法解析分数，原始输出: {}", metricName, truncate(raw, 80));
            return 0.0;
        }

        double value;
        try {
            value = Double.parseDouble(m.group(1));
        } catch (NumberFormatException e) {
            return 0.0;
        }

        // 10 分制转 0~1
        if (value > 1.0) {
            value = value / 10.0;
        }

        // 夹到 [0,1]，防止模型返回 1.5 之类的越界值污染平均值
        return Math.max(0.0, Math.min(1.0, value));
    }

    private boolean isBlank(String s) {
        return s == null || s.isBlank();
    }

    private String truncate(String s, int max) {
        if (s == null) {
            return "";
        }
        return s.length() <= max ? s : s.substring(0, max) + "...";
    }
}
