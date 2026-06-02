package com.ailearn.alibaba.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.stereotype.Service;

import java.util.List;

/**
 * <h1>第5课：AgentService — Agent 工具调用服务</h1>
 *
 * <p>展示 Agent（智能体）的核心能力：Function Calling（函数调用）。
 * Agent 能自动判断何时需要调用工具，并解析工具的返回结果。</p>
 *
 * <h2>什么是 Agent？</h2>
 * <p>Agent = LLM + 工具 + 决策能力。</p>
 * <p>普通的 Chat 只是"问什么答什么"，Agent 则能：</p>
 * <ul>
 *   <li>理解用户意图，自主决定调用哪个工具</li>
 *   <li>调用工具获取实时数据（天气、股票、数据库等）</li>
 *   <li>根据工具返回结果，继续推理或调用更多工具</li>
 *   <li>最终整合所有信息，给出完整回答</li>
 * </ul>
 *
 * <h2>ReAct 模式</h2>
 * <p>Spring AI Alibaba 的 Agent 采用 <b>ReAct</b>（Reasoning + Acting）模式：</p>
 *
 * <pre>
 *   用户: "今天杭州天气如何？123*456等于多少？"
 *
 *   ┌── Thought ── 思考 ──────────────────┐
 *   │ "用户问了天气和计算两个问题，         │
 *   │  需要调用天气工具和计算器工具"         │
 *   └─────────────────────────────────────┘
 *              │
 *     ┌────────┴────────┐
 *     ▼                 ▼
 *   Action 1          Action 2
 *   getWeather(       calculate(
 *     "杭州"            "123*456"
 *   )                 )
 *     │                 │
 *     ▼                 ▼
 *   Observation 1     Observation 2
 *   "杭州 晴 25°C"    "56088"
 *              │
 *              ▼
 *   ┌── Final Answer ─────────────────────┐
 *   │ "杭州今天晴天，气温25°C。             │
 *   │  123×456=56088"                     │
 *   └─────────────────────────────────────┘
 * </pre>
 *
 * <h2>Function Calling 本质</h2>
 * <p>大模型并不直接调用你的 Java 方法。流程如下：</p>
 * <ol>
 *   <li>Spring AI 将 @Tool 方法的信息（名称、描述、参数）发给模型</li>
 *   <li>模型判断需要调用某个工具，返回 "function_call" 指令</li>
 *   <li>Spring AI 拦截指令，在本地执行对应方法</li>
 *   <li>将执行结果返回给模型</li>
 *   <li>模型根据结果生成最终回答</li>
 * </ol>
 *
 * @author ai-learn
 * @see com.ailearn.alibaba.tool.WeatherTool
 * @see com.ailearn.alibaba.tool.CalculatorTool
 * @see com.ailearn.alibaba.tool.TimeTool
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AgentService {

    private final ChatClient chatClient;
    private final List<ToolCallback> toolCallbacks;

    /**
     * Agent 对话：自动决策并调用工具
     *
     * <h3>注册工具的方式</h3>
     * <p>Spring AI 会自动扫描所有标注了 {@code @Tool} 的方法，
     * 并将其注册为 {@link ToolCallback}。</p>
     *
     * <pre>
     * @Component
     * public class WeatherTool {
     *     @Tool(description = "查询指定城市的天气")
     *     public WeatherInfo getWeather(
     *         @ToolParam(description = "城市名称") String city
     *     ) { ... }
     * }
     * </pre>
     *
     * <h3>调用方式</h3>
     * <pre>
     * chatClient.prompt()
     *     .user(message)
     *     .tools(toolCallbacks)   // ← 注册工具
     *     .call()
     *     .content();
     * </pre>
     *
     * @param message 用户消息（可以包含多个需要工具的请求）
     * @return Agent 的完整回答
     */
    public String chat(String message) {
        log.debug("[Agent对话] 用户消息: {}", message);

        return chatClient.prompt()
                .user(message)
                // 关键：注册可用的工具列表
                // 模型会从这些工具中选择合适的来调用
                .tools(toolCallbacks.toArray(new ToolCallback[0]))
                .call()
                .content();
    }
}
