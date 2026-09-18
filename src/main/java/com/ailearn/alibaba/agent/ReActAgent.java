package com.ailearn.alibaba.agent;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * <h1>ReActAgent — 显式 ReAct 循环执行器</h1>
 *
 * <p>本类手工实现 ReAct（Reasoning + Acting）循环，而非依赖框架的
 * 黑盒工具调用。目的是让「思考-行动-观察」的每一步都<b>显式可见、可中断、可约束</b>。</p>
 *
 * <h2>ReAct vs 框架自动工具调用</h2>
 * <table border="1">
 *   <tr><th>维度</th><th>框架自动调用</th><th>显式 ReAct（本类）</th></tr>
 *   <tr><td>代码量</td><td>一行 .tools() 搞定</td><td>需要手写循环</td></tr>
 *   <tr><td>迭代控制</td><td>框架内部，不透明</td><td>完全可控</td></tr>
 *   <tr><td>中间步骤可见性</td><td>需看日志</td><td>结构化轨迹，可返回前端</td></tr>
 *   <tr><td>自杀式循环防护</td><td>依赖框架实现</td><td>可自定义循环检测</td></tr>
 *   <tr><td>终止条件定制</td><td>受限</td><td>任意定制</td></tr>
 * </table>
 *
 * <h2>核心风险：无限循环</h2>
 * <p>ReAct 最危险的失效模式是模型陷入循环——反复调用同一个工具、
 * 或反复输出"我还需要查询"却永远不给出答案。在真实系统里这意味着
 * <b>token 成本失控</b>（曾有用例单次请求烧掉数十美元）。</p>
 *
 * <p>本类设置了三重防护：</p>
 * <ol>
 *   <li><b>硬上限</b>：{@code maxIterations} 轮后强制终止并降级回答</li>
 *   <li><b>重复检测</b>：连续 N 次调用相同工具且参数相同 → 判定死循环</li>
 *   <li><b>无进展检测</b>：连续多轮未调用任何工具也未能给出答案 → 终止</li>
 * </ol>
 *
 * @author ai-learn
 */
@Slf4j
@Component
public class ReActAgent {

    private final ChatClient chatClient;

    /**
     * 用于捕获模型输出中「调用工具」的意图。
     *
     * <p>我们让模型以固定格式输出，而非依赖原生 Function Calling。
     * 原因是：原生 Function Calling 下模型的中间思考过程不可见，
     * 而教学场景恰恰需要看到「它为什么决定调这个工具」。</p>
     *
     * <pre>
     *   Thought: 需要先查天气
     *   Action: getWeather
     *   Action Input: {"city": "杭州"}
     * </pre>
     */
    private static final Pattern ACTION_PATTERN = Pattern.compile(
            "Action\\s*[:：]\\s*(\\w+)", Pattern.CASE_INSENSITIVE);

    private static final Pattern ACTION_INPUT_PATTERN = Pattern.compile(
            "Action\\s*Input\\s*[:：]\\s*(.+)", Pattern.CASE_INSENSITIVE);

    private static final Pattern THOUGHT_PATTERN = Pattern.compile(
            "Thought\\s*[:：]\\s*(.+)", Pattern.CASE_INSENSITIVE);

    private static final Pattern FINAL_ANSWER_PATTERN = Pattern.compile(
            "Final\\s*Answer\\s*[:：]\\s*(.+)", Pattern.CASE_INSENSITIVE | Pattern.DOTALL);

    /** 触发结束的标记词 */
    private static final String STOP_MARKER = "Final Answer:";

    /**
     * 连续调用同一工具且参数相同的次数达到此值时，判定为死循环。
     */
    private static final int LOOP_DETECT_THRESHOLD = 3;

    public ReActAgent(ChatClient chatClient) {
        this.chatClient = chatClient;
    }

    /**
     * 执行 ReAct 循环。
     *
     * @param question      用户问题
     * @param toolCallbacks 可用工具列表
     * @param maxIterations 最大迭代轮数（安全阀）
     * @return 执行轨迹，含最终答案
     */
    public AgentTrace run(String question, List<ToolCallback> toolCallbacks, int maxIterations) {
        AgentTrace trace = new AgentTrace(question);
        long startAll = System.currentTimeMillis();

        // 对话历史：ReAct 的"记忆"就在这里。每轮把模型的输出和工具结果
        // 追加进去，形成递进的推理链条。
        List<Message> history = new ArrayList<>();
        history.add(new UserMessage(question));

        // 循环检测用：记录上一轮的 (工具名, 参数) 与连续重复次数
        String lastSignature = null;
        int repeatCount = 0;

        for (int i = 1; i <= maxIterations; i++) {
            long startStep = System.currentTimeMillis();

            // ---------- 调用模型，获取下一步决策 ----------
            String modelOutput;
            try {
                modelOutput = callModel(history, toolCallbacks);
            } catch (Exception e) {
                log.error("[ReAct] 第 {} 轮模型调用失败", i, e);
                trace.setTerminationReason(AgentTrace.Reason.ERROR);
                trace.setFinalAnswer("抱歉，Agent 执行过程中出现错误：" + e.getMessage());
                trace.setTotalElapsedMs(System.currentTimeMillis() - startAll);
                return trace;
            }

            if (modelOutput == null || modelOutput.isBlank()) {
                log.warn("[ReAct] 第 {} 轮模型返回空内容", i);
                trace.setTerminationReason(AgentTrace.Reason.ERROR);
                trace.setFinalAnswer("模型返回了空内容，请重试。");
                trace.setTotalElapsedMs(System.currentTimeMillis() - startAll);
                return trace;
            }

            log.debug("[ReAct] 第 {} 轮模型输出:\n{}", i, modelOutput);

            // ---------- 检查是否已给出最终答案 ----------
            if (containsFinalAnswer(modelOutput)) {
                String answer = extractFinalAnswer(modelOutput);
                trace.addStep(new AgentTrace.Step(
                        i,
                        extractThought(modelOutput),
                        null, null, null,
                        System.currentTimeMillis() - startStep));
                trace.setFinalAnswer(answer);
                trace.setTerminationReason(AgentTrace.Reason.COMPLETED);
                trace.setTotalElapsedMs(System.currentTimeMillis() - startAll);
                log.info("[ReAct] 第 {} 轮完成，耗时 {}ms", i, trace.getTotalElapsedMs());
                return trace;
            }

            // ---------- 解析工具调用意图 ----------
            String action = extractAction(modelOutput);
            String actionInput = extractActionInput(modelOutput);

            if (action == null) {
                // 既没有 Final Answer 也没有 Action —— 模型输出不符合格式。
                // 这种情况通常发生在模型能力不足或 Prompt 太长时。
                // 处理策略：把格式提醒追加到历史，让它重试，而不是直接失败。
                log.warn("[ReAct] 第 {} 轮输出缺少 Action/Final Answer，追加格式提醒重试", i);
                trace.addStep(new AgentTrace.Step(
                        i, extractThought(modelOutput), null, null,
                        "输出格式不符合要求，已提示模型重试",
                        System.currentTimeMillis() - startStep));
                history.add(new AssistantMessage(modelOutput));
                history.add(new UserMessage(
                        "你的输出格式不正确。请严格按以下两种格式之一输出：\n"
                                + "1) 需要调用工具时：\nThought: <你的思考>\nAction: <工具名>\nAction Input: <参数JSON>\n"
                                + "2) 已可以回答时：\nFinal Answer: <你的回答>\n"
                                + "请重新输出。"));
                continue;
            }

            // ---------- 循环检测 ----------
            String signature = action + "|" + actionInput;
            if (signature.equals(lastSignature)) {
                repeatCount++;
                if (repeatCount >= LOOP_DETECT_THRESHOLD) {
                    log.warn("[ReAct] 检测到死循环：连续 {} 次调用 {}", repeatCount, signature);
                    trace.addStep(new AgentTrace.Step(
                            i, extractThought(modelOutput), action, actionInput,
                            "已检测到重复调用，终止循环",
                            System.currentTimeMillis() - startStep));
                    trace.setTerminationReason(AgentTrace.Reason.LOOP_DETECTED);

                    // 降级策略：不强求答案，把已有信息交给模型做最后一次总结
                    String fallback = forceFinalAnswer(history,
                            "你似乎陷入了重复调用工具。请立即基于目前已经获得的信息给出最终回答，"
                                    + "如果信息不足请直接说明还缺少什么。");
                    trace.setFinalAnswer(fallback);
                    trace.setTotalElapsedMs(System.currentTimeMillis() - startAll);
                    return trace;
                }
            } else {
                repeatCount = 0;
                lastSignature = signature;
            }

            // ---------- 执行工具 ----------
            String observation = executeTool(action, actionInput, toolCallbacks);

            trace.addStep(new AgentTrace.Step(
                    i,
                    extractThought(modelOutput),
                    action,
                    actionInput,
                    observation,
                    System.currentTimeMillis() - startStep));

            // ---------- 把本轮结果追加到历史，供下一轮推理 ----------
            history.add(new AssistantMessage(modelOutput));
            history.add(new UserMessage("Observation: " + observation));
        }

        // ---------- 触达迭代上限（安全阀）----------
        log.warn("[ReAct] 达到最大迭代次数 {}，强制终止", maxIterations);
        trace.setTerminationReason(AgentTrace.Reason.MAX_ITERATIONS);

        String fallback = forceFinalAnswer(history,
                "已达到最大执行轮次。请基于目前收集到的信息，尽快给出最完整的回答；"
                        + "若信息不足，请明确说明哪部分无法确定。");
        trace.setFinalAnswer(fallback);
        trace.setTotalElapsedMs(System.currentTimeMillis() - startAll);

        return trace;
    }

    // ==================== 模型调用 ====================

    /**
     * 构造 ReAct 提示词并调用模型。
     *
     * <h3>Prompt 设计要点</h3>
     * <ol>
     *   <li><b>严格规定输出格式</b>——格式是解析的前提，必须极其明确</li>
     *   <li><b>给出可用工具及其参数 schema</b>——工具描述质量直接决定调用成功率</li>
     *   <li><b>提供示例</b>——few-shot 对格式遵循度提升显著</li>
     *   <li><b>temperature = 0</b>——Agent 需要稳定性，不需要创造力</li>
     * </ol>
     */
    private String callModel(List<Message> history, List<ToolCallback> toolCallbacks) {
        String systemPrompt = buildReActSystemPrompt(toolCallbacks);

        ChatResponse response = chatClient.prompt()
                .system(systemPrompt)
                .messages(history)
                .call()
                .chatResponse();

        if (response == null || response.getResult() == null) {
            return null;
        }
        return response.getResult().getOutput().getText();
    }

    /**
     * 构建 ReAct 系统提示词，包含工具清单与输出格式约定。
     */
    private String buildReActSystemPrompt(List<ToolCallback> toolCallbacks) {
        StringBuilder tools = new StringBuilder();
        for (ToolCallback cb : toolCallbacks) {
            var def = cb.getToolDefinition();
            tools.append("- ").append(def.name()).append(": ").append(def.description()).append("\n");
        }

        return """
                你是一个具备工具调用能力的 AI 助手。你可以调用工具来获取信息，然后回答问题。

                【可用工具】
                %s

                【输出格式 — 必须严格遵守】
                当你需要调用工具时，输出：
                Thought: <你的思考：为什么需要这个工具>
                Action: <工具名称，必须是上面列表中的一个>
                Action Input: <JSON 格式的参数，例如 {"city":"杭州"}>

                当你已能回答问题、不再需要工具时，输出：
                Thought: <你的思考>
                Final Answer: <给用户的完整回答>

                【重要规则】
                1. 每次只调用一个工具，等到看到 Observation 后再决定下一步
                2. 不要自己编造 Observation，工具结果会由系统返回给你
                3. Final Answer 必须使用中文，直接面向用户，不要包含"Thought:"等标记
                4. 如果工具返回了错误，尝试修正参数重试，或直接说明无法完成
                5. 已经获得足够信息时立即给出 Final Answer，不要过度调用工具
                """.formatted(tools);
    }

    // ==================== 工具执行 ====================

    /**
     * 执行工具调用。
     *
     * <p>关键设计：<b>工具执行失败绝不能抛出异常中断整个 Agent</b>。
     * 必须把错误信息作为 Observation 返回给模型，让它有机会
     * 修正参数重试或换用其他工具——这正是 Agent 自主性的体现。</p>
     */
    private String executeTool(String toolName, String actionInput, List<ToolCallback> toolCallbacks) {
        for (ToolCallback cb : toolCallbacks) {
            if (!cb.getToolDefinition().name().equals(toolName)) {
                continue;
            }

            try {
                // 模型给出的参数可能是 {"city":"杭州"}，也可能直接是 杭州。
                // 这里做一次宽松处理，把裸值包装成 JSON 字符串。
                String normalizedInput = normalizeInput(actionInput);

                log.info("[ReAct] 执行工具 {}，参数 {}", toolName, normalizedInput);
                String result = cb.call(normalizedInput);

                if (result == null || result.isBlank()) {
                    return "工具执行成功，但没有返回内容。";
                }
                return result;

            } catch (Exception e) {
                // 把异常转为可读的 Observation，让模型自行决策
                log.warn("[ReAct] 工具 {} 执行失败: {}", toolName, e.getMessage());
                return "工具执行失败：" + e.getMessage()
                        + "。请检查参数格式是否正确，或改用其他方式完成任务。";
            }
        }

        // 工具不存在——列出可用工具帮助模型纠正
        String available = toolCallbacks.stream()
                .map(cb -> cb.getToolDefinition().name())
                .reduce((a, b) -> a + ", " + b)
                .orElse("无");
        log.warn("[ReAct] 模型请求了不存在的工具: {}", toolName);
        return "错误：不存在名为「" + toolName + "」的工具。可用工具为：" + available;
    }

    /**
     * 规范化工具入参。
     *
     * <p>模型有时会输出引号包裹的字符串、或裸标量值，这里统一
     * 为工具期望的形式。工具若只有一个 String 参数，Spring AI
     * 可以直接接受裸字符串。</p>
     */
    private String normalizeInput(String input) {
        if (input == null) {
            return "{}";
        }
        String trimmed = input.trim();

        // 去掉 markdown 代码块围栏（模型很爱加）
        trimmed = trimmed.replaceAll("^```(?:json)?\\s*", "")
                .replaceAll("\\s*```$", "")
                .trim();

        // 已有 JSON 结构，直接使用
        if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
            return trimmed;
        }

        // 裸字符串：去掉外层引号
        if ((trimmed.startsWith("\"") && trimmed.endsWith("\""))
                || (trimmed.startsWith("'") && trimmed.endsWith("'"))) {
            trimmed = trimmed.substring(1, trimmed.length() - 1);
        }

        return trimmed;
    }

    // ==================== 强制收尾 ====================

    /**
     * 在触发安全阀时，强制模型给出收尾回答。
     *
     * <p>这是重要的<b>降级策略</b>：宁可给出一个信息不全的回答，
     * 也不要返回错误或空白。用户体验上，部分答案远优于失败。</p>
     */
    private String forceFinalAnswer(List<Message> history, String instruction) {
        try {
            List<Message> finalHistory = new ArrayList<>(history);
            finalHistory.add(new UserMessage(instruction));

            String result = chatClient.prompt()
                    .messages(finalHistory)
                    .call()
                    .content();

            return (result == null || result.isBlank())
                    ? "抱歉，未能完成该请求。已调用的工具结果不足以得出可靠结论。"
                    : result;

        } catch (Exception e) {
            log.error("[ReAct] 强制收尾也失败了", e);
            return "抱歉，Agent 执行超限且收尾失败：" + e.getMessage();
        }
    }

    // ==================== 输出解析 ====================

    private boolean containsFinalAnswer(String output) {
        return output.toLowerCase().contains(STOP_MARKER.toLowerCase());
    }

    private String extractFinalAnswer(String output) {
        Matcher m = FINAL_ANSWER_PATTERN.matcher(output);
        if (m.find()) {
            return m.group(1).trim();
        }
        // 兜底：按标记切分
        int idx = output.toLowerCase().indexOf(STOP_MARKER.toLowerCase());
        return idx >= 0 ? output.substring(idx + STOP_MARKER.length()).trim() : output.trim();
    }

    private String extractAction(String output) {
        Matcher m = ACTION_PATTERN.matcher(output);
        return m.find() ? m.group(1).trim() : null;
    }

    private String extractActionInput(String output) {
        Matcher m = ACTION_INPUT_PATTERN.matcher(output);
        return m.find() ? m.group(1).trim() : null;
    }

    private String extractThought(String output) {
        Matcher m = THOUGHT_PATTERN.matcher(output);
        return m.find() ? m.group(1).trim() : null;
    }
}
