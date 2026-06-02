package com.ailearn.alibaba.controller;

import com.ailearn.alibaba.model.ChatRequest;
import com.ailearn.alibaba.service.MemoryChatService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.UUID;

/**
 * <h1>MemoryChatController — 带记忆的对话接口</h1>
 *
 * <p>支持多轮对话，AI 能记住之前的聊天内容。</p>
 *
 * <h2>使用方式</h2>
 * <pre>
 * // 第一轮：系统自动分配 conversationId
 * POST /api/memory/chat
 * { "message": "我叫张三" }
 * → { "conversationId": "abc-123", "content": "你好张三！" }
 *
 * // 第二轮：传入相同的 conversationId
 * POST /api/memory/chat
 * { "message": "我叫什么？", "conversationId": "abc-123" }
 * → { "conversationId": "abc-123", "content": "你叫张三" }
 * </pre>
 *
 * @author ai-learn
 */
@Slf4j
@RestController
@RequestMapping("/api/memory")
@RequiredArgsConstructor
public class MemoryChatController {

    private final MemoryChatService memoryChatService;

    /**
     * 带记忆的多轮对话
     *
     * <p>如果 conversationId 为空，系统自动生成一个新的。</p>
     */
    @PostMapping("/chat")
    public Map<String, String> chat(@RequestBody ChatRequest request) {
        // 如果没传 conversationId，自动生成一个
        String conversationId = request.getConversationId();
        if (conversationId == null || conversationId.isBlank()) {
            conversationId = UUID.randomUUID().toString().substring(0, 8);
            log.info("[记忆对话] 新会话: {}", conversationId);
        }

        String content = memoryChatService.chat(request.getMessage(), conversationId);

        return Map.of(
                "conversationId", conversationId,
                "content", content
        );
    }
}
