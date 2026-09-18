package com.ailearn.alibaba.agent;

import java.util.ArrayList;
import java.util.List;

/**
 * <h1>AgentTrace — Agent 执行轨迹</h1>
 *
 * <p>记录一次 Agent 调用的完整决策过程。这是 Agent 可观测性的基础。</p>
 *
 * <h2>为什么 Agent 必须可观测？</h2>
 * <p>Agent 与普通 Chat 最大的工程差异在于：<b>它有中间步骤</b>。
 * 一个 Agent 回答错了，可能是：</p>
 * <ul>
 *   <li>没选对工具 → 工具描述有问题</li>
 *   <li>选对了工具但参数错 → 参数 schema 描述不清</li>
 *   <li>工具执行失败 → 工具实现有 bug</li>
 *   <li>拿到了正确结果但推理错 → 模型能力或 Prompt 问题</li>
 * </ul>
 *
 * <p>如果没有轨迹，这四种情况在外部看到的<b>现象完全一样</b>——
 * 都是「回答不对」。排查只能靠猜。有了轨迹，可以直接定位到出错的环节。</p>
 *
 * <h2>数据结构</h2>
 * <pre>
 *   AgentTrace
 *     ├─ step 1: Thought "需要查询杭州天气"
 *     │           Action  getWeather(city="杭州")
 *     │           Observation "杭州 晴 25°C"
 *     ├─ step 2: Thought "还需要计算"
 *     │           Action  calculate(expression="123*456")
 *     │           Observation "123*456 = 56088.00"
 *     └─ final:  "杭州今天晴，25°C；123×456 = 56088"
 * </pre>
 *
 * @author ai-learn
 */
public class AgentTrace {

    /**
     * 单个 ReAct 步骤。
     *
     * @param iteration   第几轮迭代（从 1 开始）
     * @param thought     模型的思考内容
     * @param action      决定调用的工具名（未调用工具时为 null）
     * @param actionInput 工具入参（原始字符串）
     * @param observation 工具返回结果
     * @param elapsedMs   本轮耗时
     */
    public record Step(
            int iteration,
            String thought,
            String action,
            String actionInput,
            String observation,
            long elapsedMs
    ) {
        /** 该步骤是否真的调用了工具 */
        public boolean hasToolCall() {
            return action != null && !action.isBlank();
        }
    }

    private final String question;
    private final List<Step> steps = new ArrayList<>();
    private String finalAnswer;
    private String terminationReason = "UNKNOWN";
    private long totalElapsedMs;

    public AgentTrace(String question) {
        this.question = question;
    }

    public void addStep(Step step) {
        steps.add(step);
    }

    // ---------- getter / setter ----------

    public String getQuestion() {
        return question;
    }

    public List<Step> getSteps() {
        return steps;
    }

    public String getFinalAnswer() {
        return finalAnswer;
    }

    public void setFinalAnswer(String finalAnswer) {
        this.finalAnswer = finalAnswer;
    }

    public String getTerminationReason() {
        return terminationReason;
    }

    public void setTerminationReason(String terminationReason) {
        this.terminationReason = terminationReason;
    }

    public long getTotalElapsedMs() {
        return totalElapsedMs;
    }

    public void setTotalElapsedMs(long totalElapsedMs) {
        this.totalElapsedMs = totalElapsedMs;
    }

    /**
     * 工具调用总次数。
     *
     * <p>可用于监控异常：单次请求工具调用超过 5 次通常意味着
     * 模型陷入了循环，或工具描述让它误以为需要反复尝试。</p>
     */
    public int toolCallCount() {
        return (int) steps.stream().filter(Step::hasToolCall).count();
    }

    /**
     * 输出人类可读的轨迹，便于日志排查。
     */
    public String render() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Agent 执行轨迹 ===\n");
        sb.append("问题: ").append(question).append("\n");
        sb.append("终止原因: ").append(terminationReason)
                .append(" | 迭代 ").append(steps.size())
                .append(" 轮 | 工具调用 ").append(toolCallCount())
                .append(" 次 | 总耗时 ").append(totalElapsedMs).append("ms\n");

        for (Step s : steps) {
            sb.append("\n--- Step ").append(s.iteration()).append(" (").append(s.elapsedMs()).append("ms) ---\n");
            if (s.thought() != null && !s.thought().isBlank()) {
                sb.append("Thought:     ").append(truncate(s.thought(), 200)).append("\n");
            }
            if (s.hasToolCall()) {
                sb.append("Action:      ").append(s.action()).append("\n");
                sb.append("ActionInput: ").append(truncate(s.actionInput(), 200)).append("\n");
                sb.append("Observation: ").append(truncate(s.observation(), 300)).append("\n");
            }
        }

        sb.append("\n最终回答: ").append(truncate(finalAnswer, 500)).append("\n");
        return sb.toString();
    }

    private String truncate(String s, int max) {
        if (s == null) {
            return "(null)";
        }
        String flat = s.replaceAll("\\s+", " ").trim();
        return flat.length() <= max ? flat : flat.substring(0, max) + "...";
    }

    /** 终止原因常量 */
    public static final class Reason {
        /** 模型给出了最终答案，正常结束 */
        public static final String COMPLETED = "COMPLETED";
        /** 达到最大迭代次数，强制结束（安全阀触发） */
        public static final String MAX_ITERATIONS = "MAX_ITERATIONS";
        /** 模型连续重复调用同一工具，判定为陷入循环 */
        public static final String LOOP_DETECTED = "LOOP_DETECTED";
        /** 执行过程抛出异常 */
        public static final String ERROR = "ERROR";

        private Reason() {}
    }
}
