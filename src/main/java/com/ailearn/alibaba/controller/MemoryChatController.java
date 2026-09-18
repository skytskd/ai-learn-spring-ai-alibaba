package com.ailearn.alibaba.controller;

import com.ailearn.alibaba.memory.ConversationState;
import com.ailearn.alibaba.memory.LayeredMemoryService;
import com.ailearn.alibaba.memory.MemoryMessage;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * <h1>MemoryChatController（强化版）— 分层记忆对话接口</h1>
 *
 * <p>相比原版只支持「窗口内记住」，现在支持三层记忆的完整能力，
 * 并暴露接口让你能看到记忆内部状态。</p>
 *
 * <h2>接口列表</h2>
 * <table border="1">
 *   <tr><th>接口</th><th>用途</th></tr>
 *   <tr><td>POST /api/memory/chat</td><td>多轮对话（自动生成/复用会话 ID）</td></tr>
 *   <tr><td>GET  /api/memory/state</td><td>查看某会话的三层记忆状态</td></tr>
 *   <tr><td>DELETE /api/memory/clear</td><td>清空某会话记忆</td></tr>
 *   <tr><td>GET  /api/memory/stats</td><td>查看记忆系统总体统计</td></tr>
 *   <tr><td>GET  /api/memory/persistence-guide</td><td>生产环境持久化方案说明</td></tr>
 * </table>
 *
 * <h2>为什么要把记忆状态暴露出来？</h2>
 * <p>多轮记忆最容易出的问题是「它到底记住了什么」。如果只能
 * 通过「问模型它还记不记得」来验证，那是在用一个不确定的系统
 * 去测试另一个不确定的系统。直接暴露内部状态，才能确认
 * 摘要是否合理、事实抽取是否准确。</p>
 *
 * @author ai-learn
 */
@Slf4j
@RestController
@RequestMapping("/api/memory")
@RequiredArgsConstructor
public class MemoryChatController {

    private final LayeredMemoryService memoryService;

    /**
     * 带分层记忆的多轮对话。
     *
     * <p>返回的 {@code meta} 字段包含本轮的记忆状态快照，
     * 便于前端展示「记忆进度」或调试。</p>
     */
    @PostMapping("/chat")
    public Map<String, Object> chat(@RequestBody Map<String, String> request) {
        String message = request.get("message");
        String conversationId = request.get("conversationId");

        if (message == null || message.isBlank()) {
            return Map.of("error", "message 不能为空");
        }

        if (conversationId == null || conversationId.isBlank()) {
            conversationId = UUID.randomUUID().toString().substring(0, 8);
            log.info("[记忆对话] 创建新会话: {}", conversationId);
        }

        LayeredMemoryService.MemoryChatResult result =
                memoryService.chat(conversationId, message);

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("conversationId", result.conversationId());
        response.put("content", result.reply());

        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("windowSize", result.windowSize());
        meta.put("summarizedThisTurn", result.summarized());
        meta.put("longTermFacts", result.longTermFacts());
        meta.put("elapsedMs", result.elapsedMs());
        response.put("meta", meta);

        return response;
    }

    /**
     * 查看某会话的完整记忆状态。
     *
     * <p>这是调试分层记忆的主要手段：可以直观看到
     * 摘要写了什么、抽取了哪些事实、窗口里还剩哪些消息。</p>
     */
    @GetMapping("/state")
    public Map<String, Object> state(@RequestParam String conversationId) {
        ConversationState state = memoryService.getState(conversationId);

        Map<String, Object> result = new LinkedHashMap<>();
        if (state == null) {
            result.put("exists", false);
            result.put("message", "会话不存在或已过期");
            return result;
        }

        result.put("exists", true);
        result.put("conversationId", state.getConversationId());
        result.put("totalTurns", state.getTotalTurns());

        // Layer 1
        List<Map<String, String>> window = new ArrayList<>();
        for (MemoryMessage m : state.getRecentMessages()) {
            Map<String, String> item = new LinkedHashMap<>();
            item.put("role", m.getRole().name());
            item.put("content", m.getContent() == null || m.getContent().length() <= 200
                    ? m.getContent()
                    : m.getContent().substring(0, 200) + "...");
            window.add(item);
        }
        result.put("layer1_window", window);
        result.put("layer1_size", window.size());

        // Layer 2
        result.put("layer2_summary", state.getSummary() == null || state.getSummary().isBlank()
                ? "(尚未生成摘要)" : state.getSummary());

        // Layer 3
        result.put("layer3_longTermFacts", state.getLongTermFacts());

        return result;
    }

    /**
     * 清空指定会话的记忆。
     */
    @DeleteMapping("/clear")
    public Map<String, Object> clear(@RequestParam String conversationId) {
        boolean removed = memoryService.clear(conversationId);
        log.info("[记忆对话] 清空会话 {}: {}", conversationId, removed ? "成功" : "不存在");
        return Map.of(
                "conversationId", conversationId,
                "cleared", removed);
    }

    /**
     * 记忆系统总体统计。
     */
    @GetMapping("/stats")
    public Map<String, Object> stats() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("activeConversations", memoryService.activeConversations());
        result.put("storageType", "进程内 ConcurrentHashMap（重启丢失）");
        result.put("note", "生产环境应替换为 Redis + 数据库冷归档");
        return result;
    }

    /**
     * 生产环境持久化方案说明。
     */
    @GetMapping("/persistence-guide")
    public Map<String, String> persistenceGuide() {
        return Map.of("guide", memoryService.describePersistenceOptions());
    }
}
