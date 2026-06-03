package com.ailearn.alibaba.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.stereotype.Service;

/**
 * <h1>第3课：MemoryChatService — 对话记忆服务</h1>
 *
 * <p>展示如何让 AI 记住上下文，实现多轮对话。</p>
 *
 * @author ai-learn
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class MemoryChatService {

    private final ChatClient chatClient;

    /**
     * 对话记忆存储（基于内存 Map，限制最多保留 20 条消息）
     *
     * <p>Spring AI 1.1.0 中，InMemoryChatMemory 被 MessageWindowChatMemory 替代。
     * MessageWindowChatMemory 内部使用 InMemoryChatMemoryRepository 存储数据，
     * 并通过 maxMessages 参数自动淘汰旧消息，防止内存溢出。</p>
     */
    private final ChatMemory chatMemory = MessageWindowChatMemory.builder()
            .maxMessages(20)
            .build();

    /**
     * 带记忆的多轮对话
     *
     * <p>Spring AI 1.1.0 中 MessageChatMemoryAdvisor 的 API 变化：
     * 不再通过 advisorSpec.param() 传参，而是通过 builder 构建 Advisor 实例。</p>
     *
     * @param message        用户输入
     * @param conversationId 对话ID（相同ID共享记忆）
     * @return AI 回复内容
     */
    public String chat(String message, String conversationId) {
        log.debug("[记忆对话] conversationId={}, message={}", conversationId, message);

        // Spring AI 1.1.0：使用 builder 创建 MessageChatMemoryAdvisor
        MessageChatMemoryAdvisor advisor = MessageChatMemoryAdvisor.builder(chatMemory)
                .conversationId(conversationId)
                .build();

        return chatClient.prompt()
                .user(message)
                .advisors(advisor)
                .call()
                .content();
    }
}
