package com.ailearn.alibaba.controller;

import com.ailearn.alibaba.model.ChatRequest;
import com.ailearn.alibaba.service.AgentService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * <h1>AgentController — Agent 工具调用接口</h1>
 *
 * <p>Agent 能自动调用工具（天气查询、计算器等）来完成复杂任务。</p>
 *
 * <h2>测试示例</h2>
 * <pre>
 * curl -X POST http://localhost:8080/api/agent/chat \
 *   -H "Content-Type: application/json" \
 *   -d '{"message":"今天杭州天气如何？123*456等于多少？"}'
 * </pre>
 *
 * <h2>Agent 的决策过程</h2>
 * <ol>
 *   <li>理解用户意图 → "需要天气和计算两个工具"</li>
 *   <li>调用 getWeather("杭州") → 获取天气数据</li>
 *   <li>调用 calculate("123*456") → 获取计算结果</li>
 *   <li>整合信息 → 生成自然语言回答</li>
 * </ol>
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
     * Agent 对话接口
     *
     * <p>用户可以用自然语言描述需求，Agent 会自动选择合适的工具。</p>
     */
    @PostMapping("/chat")
    public Map<String, String> chat(@RequestBody ChatRequest request) {
        log.info("[Agent对话] {}", request.getMessage());
        String content = agentService.chat(request.getMessage());
        return Map.of("content", content);
    }
}
