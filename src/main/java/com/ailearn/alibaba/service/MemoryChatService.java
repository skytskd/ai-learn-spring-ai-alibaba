package com.ailearn.alibaba.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.InMemoryChatMemory;
import org.springframework.stereotype.Service;

/**
 * <h1>第3课：MemoryChatService — 对话记忆服务</h1>
 *
 * <p>展示如何让 AI 记住上下文，实现多轮对话。</p>
 *
 * <h2>为什么需要对话记忆？</h2>
 * <p>每次调用 AI API 都是 <b>无状态</b> 的——模型不知道"上一轮聊了什么"。
 * 要实现多轮对话，必须把历史消息一起发送。</p>
 *
 * <pre>
 * 无记忆（每次都是新对话）：
 * 用户: "我叫张三"    → AI: "你好张三！"
 * 用户: "我叫什么？"  → AI: "我不知道..."  ❌
 *
 * 有记忆（携带历史消息）：
 * 用户: "我叫张三"    → AI: "你好张三！"
 * 用户: "我叫什么？"  → AI: "你叫张三"    ✅
 * </pre>
 *
 * <h2>核心概念：Advisor 模式</h2>
 *
 * <h3>什么是 Advisor？</h3>
 * <p>Advisor 是 Spring AI 中的 AOP 机制。在请求发送到模型之前/之后执行拦截逻辑。
 * 类似于 Spring MVC 的 Interceptor 或 Servlet 的 Filter。</p>
 *
 * <pre>
 * 请求 ──→ [Advisor1] ──→ [Advisor2] ──→ [Advisor3] ──→ ChatModel
 *            │               │               │
 *            日志记录        记忆注入         安全检查
 * </pre>
 *
 * <h3>MessageChatMemoryAdvisor</h3>
 * <p>专门处理对话记忆的 Advisor，工作原理：</p>
 * <ol>
 *   <li><b>请求前</b>：从 ChatMemory 读取历史，注入到 Prompt 中</li>
 *   <li><b>请求后</b>：将本轮对话（用户消息 + AI 回复）存入 ChatMemory</li>
 * </ol>
 *
 * <h3>ChatMemory 接口</h3>
 * <p>Spring AI 提供了多种实现：</p>
 * <table border="1">
 *   <tr><th>实现</th><th>存储</th><th>适用场景</th></tr>
 *   <tr><td>InMemoryChatMemory</td><td>内存 Map</td><td>学习/开发</td></tr>
 *   <tr><td>CassandraChatMemory</td><td>Cassandra</td><td>生产环境</td></tr>
 *   <tr><td>JdbcChatMemory</td><td>数据库</td><td>持久化</td></tr>
 * </table>
 *
 * @author ai-learn
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class MemoryChatService {

    private final ChatClient chatClient;

    /**
     * 对话记忆存储（基于内存 Map）
     *
     * <p>Key = conversationId，Value = 该对话的消息列表。
     * 生产环境应替换为持久化实现（如 Redis、数据库）。</p>
     */
    private final ChatMemory chatMemory = new InMemoryChatMemory();

    /**
     * 带记忆的多轮对话
     *
     * <h3>执行流程</h3>
     * <ol>
     *   <li>创建 MessageChatMemoryAdvisor，绑定 chatMemory</li>
     *   <li>通过 .advisors() 链式注入到 ChatClient</li>
     *   <li>每次调用时传入 conversationId 区分不同对话</li>
     *   <li>Advisor 自动完成"读历史 → 注入 → 存历史"</li>
     * </ol>
     *
     * <h3>conversationId 的作用</h3>
     * <p>用于隔离不同用户的对话上下文：</p>
     * <ul>
     *   <li>conversationId = "user-001" → 用户A的对话历史</li>
     *   <li>conversationId = "user-002" → 用户B的对话历史</li>
     * </ul>
     *
     * <h3>内存管理注意</h3>
     * <p>InMemoryChatMemory 默认不限制消息数。长时间运行会导致 OOM。
     * 可通过自定义 ChatMemory 实现来限制（如 LRU 淘汰）。</p>
     *
     * @param message        用户输入
     * @param conversationId 对话ID（相同ID共享记忆）
     * @return AI 回复内容
     */
    public String chat(String message, String conversationId) {
        log.debug("[记忆对话] conversationId={}, message={}", conversationId, message);

        return chatClient.prompt()
                .user(message)
                // 注入记忆 Advisor
                // 关键参数：
                //   - chatMemory：存储对话历史
                //   - conversationId：对话标识
                //   - chatMemoryRetrieveSize：每次携带的历史消息数（默认100）
                .advisors(advisorSpec -> advisorSpec
                        .param(MessageChatMemoryAdvisor.CHAT_MEMORY_CONVERSATION_ID_KEY, conversationId)
                        .param(MessageChatMemoryAdvisor.CHAT_MEMORY_RETRIEVE_SIZE_KEY, 20)
                )
                .call()
                .content();
    }
}
