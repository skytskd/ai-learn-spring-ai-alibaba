package com.ailearn.alibaba.service;

import com.ailearn.alibaba.agent.AgentTrace;
import com.ailearn.alibaba.agent.ReActAgent;
import com.ailearn.alibaba.config.AgentProperties;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * <h1>第5课（强化版）：AgentService — Agent 编排服务</h1>
 *
 * <p>提供两种 Agent 执行模式，便于对比理解编排策略的取舍。</p>
 *
 * <h2>模式对比</h2>
 *
 * <h3>模式 A：框架原生 Function Calling（{@link #chatSimple}）</h3>
 * <pre>
 *   chatClient.prompt().user(msg).tools(callbacks).call().content()
 * </pre>
 * <p>一行代码完成。框架内部自动处理「模型请求工具 → 执行 → 回传 → 再请求」的循环。
 * 优点：代码极简。缺点：迭代次数不可控、中间过程不可见、无法定制终止条件。</p>
 *
 * <h3>模式 B：显式 ReAct 循环（{@link #chatWithTrace}）</h3>
 * <pre>
 *   手写 Thought → Action → Observation 循环
 *   三重安全阀：最大迭代 / 重复调用检测 / 格式纠错重试
 * </pre>
 * <p>优点：完全可控、全程可观测、成本可预测。
 * 缺点：代码量大，且依赖模型对输出格式的遵循能力。</p>
 *
 * <h2>选型建议</h2>
 * <table border="1">
 *   <tr><th>场景</th><th>推荐模式</th><th>理由</th></tr>
 *   <tr><td>学习 / 演示</td><td>模式 B</td><td>能看清 Agent 的每个决策环节</td></tr>
 *   <tr><td>简单固定任务</td><td>模式 A</td><td>工具少、任务明确，不需要额外控制</td></tr>
 *   <tr><td>生产环境 / 成本敏感</td><td>模式 B</td><td>需要硬性成本上限与可观测性</td></tr>
 *   <tr><td>复杂多步任务</td><td>模式 B</td><td>需要精细的终止与降级策略</td></tr>
 * </table>
 *
 * <h2>为什么工具必须"小"？</h2>
 * <p>工具数量与描述质量直接影响调用成功率：</p>
 * <ul>
 *   <li><b>数量</b>：超过 10 个工具后，模型选错的概率显著上升
 *       （工具描述在 prompt 中互相干扰）。推荐控制在 5 ~ 8 个。</li>
 *   <li><b>描述</b>：描述必须说明「什么时候用」，而非只说「做什么」。
 *       反例："查询用户信息"。正例："根据用户 ID 查询用户的姓名、
 *       邮箱和注册时间。当用户询问自己的账号信息时使用。"</li>
 *   <li><b>粒度</b>：一个工具做一件事。若一个工具既查天气又查股票，
 *       模型很难决定何时调用它。</li>
 * </ul>
 *
 * @author ai-learn
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AgentService {

    private final ChatClient chatClient;
    private final List<ToolCallback> toolCallbacks;
    private final ReActAgent reActAgent;
    private final AgentProperties agentProperties;

    /**
     * 模式 A：框架原生工具调用（保留用于对比教学）。
     *
     * @param message 用户消息
     * @return Agent 回答
     */
    public String chatSimple(String message) {
        log.debug("[Agent-简单模式] 用户消息: {}", message);

        return chatClient.prompt()
                .user(message)
                // 显式传数组并加 (ToolCallback[]) 转换，避免 varargs 的
                // "可能有副作用的泛型数组创建" 警告
                .tools((ToolCallback[]) toolCallbacks.toArray(new ToolCallback[0]))
                .call()
                .content();
    }

    /**
     * 模式 B：显式 ReAct 循环，返回完整执行轨迹。
     *
     * <p>这是推荐的调用方式，具备：</p>
     * <ul>
     *   <li>硬性迭代上限（{@code ai-learn.agent.max-iterations}）</li>
     *   <li>死循环检测</li>
     *   <li>格式纠错重试</li>
     *   <li>超限降级回答</li>
     *   <li>结构化轨迹输出</li>
     * </ul>
     *
     * @param message 用户消息
     * @return 含轨迹的执行结果
     */
    public AgentTrace chatWithTrace(String message) {
        log.info("[Agent-ReAct模式] 用户消息: {}，最大迭代 {}", message, agentProperties.getMaxIterations());

        AgentTrace trace = reActAgent.run(
                message,
                toolCallbacks,
                agentProperties.getMaxIterations());

        log.info("[Agent-ReAct模式] 完成，终止原因={}，迭代={}，工具调用={}，耗时={}ms",
                trace.getTerminationReason(),
                trace.getSteps().size(),
                trace.toolCallCount(),
                trace.getTotalElapsedMs());

        return trace;
    }

    /**
     * 统一入口：按配置决定使用哪种模式，只返回文本答案。
     *
     * @param message 用户消息
     * @return 文本答案
     */
    public String chat(String message) {
        if (!agentProperties.isEnableReAct()) {
            return chatSimple(message);
        }

        AgentTrace trace = chatWithTrace(message);

        if (agentProperties.isReturnTrace()) {
            // 附加一行执行摘要，便于在简单调用时也能看出 Agent 做了什么
            return trace.getFinalAnswer()
                    + "\n\n---\n*执行信息：迭代 "
                    + trace.getSteps().size() + " 轮，工具调用 "
                    + trace.toolCallCount() + " 次，耗时 "
                    + trace.getTotalElapsedMs() + "ms，终止原因 "
                    + trace.getTerminationReason() + "*";
        }

        return trace.getFinalAnswer();
    }

    /** 当前注册的可用工具数量，供健康检查与调试接口使用 */
    public int availableToolCount() {
        return toolCallbacks.size();
    }

    /** 列出全部可用工具，便于前端展示 Agent 能力边界 */
    public List<String> listTools() {
        return toolCallbacks.stream()
                .map(cb -> cb.getToolDefinition().name()
                        + " — " + cb.getToolDefinition().description())
                .toList();
    }
}
