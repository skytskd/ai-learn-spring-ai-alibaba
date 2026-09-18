package com.ailearn.alibaba.controller;

import com.ailearn.alibaba.agent.AgentTrace;
import com.ailearn.alibaba.model.ChatRequest;
import com.ailearn.alibaba.service.AgentService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * <h1>AgentController（强化版）— Agent 编排接口</h1>
 *
 * <h2>接口列表</h2>
 * <table border="1">
 *   <tr><th>接口</th><th>说明</th></tr>
 *   <tr><td>POST /api/agent/chat</td>
 *       <td>统一入口，按配置走 ReAct 或原生模式</td></tr>
 *   <tr><td>POST /api/agent/chat-trace</td>
 *       <td>ReAct 模式，返回完整执行轨迹（结构化）</td></tr>
 *   <tr><td>POST /api/agent/chat-simple</td>
 *       <td>框架原生工具调用（对照用）</td></tr>
 *   <tr><td>GET  /api/agent/tools</td>
 *       <td>列出当前注册的全部工具</td></tr>
 * </table>
 *
 * <h2>轨迹为什么重要？</h2>
 * <p>Agent 回答错误时，没有轨迹你只能看到「答案不对」这一个现象。
 * 有轨迹后能直接看到：它想到了什么、调了哪个工具、传了什么参数、
 * 工具返回了什么、在哪一步偏离。</p>
 *
 * <p>本接口返回的结构化轨迹是排查 Agent 问题的第一手材料。</p>
 *
 * @author ai-learn
 */
@Slf4j
@RestController
@RequestMapping("/api/agent")
@RequiredArgsConstructor
public class AgentController {

    private final AgentService agentService;

    /**
     * 统一入口。
     */
    @PostMapping("/chat")
    public Map<String, String> chat(@RequestBody ChatRequest request) {
        log.info("[Agent对话] {}", request.getMessage());
        String content = agentService.chat(request.getMessage());
        return Map.of("content", content);
    }

    /**
     * ReAct 模式：返回结构化执行轨迹。
     *
     * <h3>返回内容</h3>
     * <ul>
     *   <li>{@code answer} — 最终回答</li>
     *   <li>{@code steps} — 每一步的 Thought / Action / ActionInput / Observation</li>
     *   <li>{@code terminationReason} — 终止原因</li>
     *   <li>{@code toolCallCount} — 工具调用次数</li>
     *   <li>{@code elapsedMs} — 总耗时</li>
     * </ul>
     */
    @PostMapping("/chat-trace")
    public Map<String, Object> chatTrace(@RequestBody ChatRequest request) {
        log.info("[Agent-ReAct] {}", request.getMessage());

        AgentTrace trace = agentService.chatWithTrace(request.getMessage());

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("question", trace.getQuestion());
        result.put("answer", trace.getFinalAnswer());
        result.put("terminationReason", trace.getTerminationReason());
        result.put("terminationMeaning", explainReason(trace.getTerminationReason()));
        result.put("iterationCount", trace.getSteps().size());
        result.put("toolCallCount", trace.toolCallCount());
        result.put("elapsedMs", trace.getTotalElapsedMs());

        List<Map<String, Object>> steps = new ArrayList<>();
        for (AgentTrace.Step s : trace.getSteps()) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("iteration", s.iteration());
            item.put("thought", s.thought());
            item.put("action", s.action());
            item.put("actionInput", s.actionInput());
            item.put("observation", s.observation());
            item.put("elapsedMs", s.elapsedMs());
            steps.add(item);
        }
        result.put("steps", steps);

        return result;
    }

    /**
     * 框架原生工具调用（对照用）。
     */
    @PostMapping("/chat-simple")
    public Map<String, String> chatSimple(@RequestBody ChatRequest request) {
        log.info("[Agent-原生模式] {}", request.getMessage());
        String content = agentService.chatSimple(request.getMessage());
        return Map.of("content", content);
    }

    /**
     * 列出可用工具。
     *
     * <p>工具清单直接决定了 Agent 的能力边界。
     * 这个接口便于确认工具是否被正确注册。</p>
     */
    @GetMapping("/tools")
    public Map<String, Object> tools() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("count", agentService.availableToolCount());
        result.put("tools", agentService.listTools());
        result.put("note", "建议工具数量控制在 5~8 个；描述应说明「何时使用」而非仅「做什么」");
        return result;
    }

    /**
     * 把终止原因翻译成人话，便于前端展示与快速定位问题。
     */
    private String explainReason(String reason) {
        return switch (reason) {
            case AgentTrace.Reason.COMPLETED -> "模型给出了最终答案，正常结束";
            case AgentTrace.Reason.MAX_ITERATIONS -> "达到最大迭代次数被强制中断，"
                    + "说明任务可能需要更多步骤，或模型陷入了无效循环";
            case AgentTrace.Reason.LOOP_DETECTED -> "检测到重复调用同一工具，"
                    + "已主动终止以避免成本失控";
            case AgentTrace.Reason.ERROR -> "执行过程中发生错误";
            default -> "未知原因";
        };
    }
}
