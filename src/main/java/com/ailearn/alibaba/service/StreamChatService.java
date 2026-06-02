package com.ailearn.alibaba.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;

/**
 * <h1>第2课：StreamChatService — 流式对话服务</h1>
 *
 * <p>展示如何实现 AI 的流式输出（逐字返回，类似 ChatGPT 的打字效果）。</p>
 *
 * <h2>为什么需要流式输出？</h2>
 * <ul>
 *   <li><b>用户体验</b>：用户不需要等待完整回答，看到内容逐字出现</li>
 *   <li><b>长文本</b>：对于长回答，流式输出避免长时间白屏</li>
 *   <li><b>实时交互</b>：适合聊天场景，让对话更自然</li>
 * </ul>
 *
 * <h2>技术原理</h2>
 * <h3>SSE（Server-Sent Events）</h3>
 * <p>SSE 是 HTTP 协议的一种扩展，服务器可以向客户端推送数据流。
 * 客户端通过 EventSource API 接收数据。</p>
 *
 * <pre>
 * 客户端                        服务端
 *   |                            |
 *   |-- GET /api/chat/stream --->|
 *   |                            |-- 处理请求
 *   |<-- data: "你" ------------|
 *   |<-- data: "好" ------------|
 *   |<-- data: "，" ------------|
 *   |<-- data: "我" ------------|
 *   |<-- data: "是" ------------|
 *   |<-- data: "AI" ------------|
 *   |<-- data: [DONE] ----------|
 * </pre>
 *
 * <h3>Flux（响应式流）</h3>
 * <p>Spring WebFlux 的 Flux 是 Reactive Streams 规范的实现，
 * 表示一个异步的 0..N 个元素的数据流。</p>
 *
 * <pre>
 * Flux<String>                    Flux&lt;ChatResponse&gt;
 * 数据流                         响应流
 * ┌─────┐                       ┌──────────┐
 * │ "你" │──→ map ──→           │ ChatResp │──→ SSE ──→ 浏览器
 * │ "好" │                       │ ChatResp │
 * │ "，" │                       │ ChatResp │
 * │ ... │                       │   ...    │
 * └─────┘                       └──────────┘
 * </pre>
 *
 * <h2>关键区别</h2>
 * <table border="1">
 *   <tr><th></th><th>chatClient.call()</th><th>chatClient.stream()</th></tr>
 *   <tr><td>返回值</td><td>ChatResponse（阻塞）</td><td>Flux&lt;String&gt;（非阻塞）</td></tr>
 *   <tr><td>返回方式</td><td>一次性返回全文</td><td>逐 Token 返回</td></tr>
 *   <tr><td>用户感知</td><td>等待后全部出现</td><td>逐字打字效果</td></tr>
 *   <tr><td>适用场景</td><td>API 调用、批处理</td><td>聊天、实时交互</td></tr>
 * </table>
 *
 * @author ai-learn
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class StreamChatService {

    private final ChatClient chatClient;

    /**
     * 流式对话：逐 Token 返回 AI 回复
     *
     * <h3>执行流程</h3>
     * <ol>
     *   <li>调用 {@code chatClient.prompt().user(message).stream().content()}</li>
     *   <li>底层通过 DashScope API 的 SSE 接口逐 Token 获取数据</li>
     *   <li>每个 Token 作为一个字符串元素发射到 Flux 流中</li>
     *   <li>前端通过 EventSource 逐条接收并显示</li>
     * </ol>
     *
     * <h3>Flux 说明</h3>
     * <p>Flux 是响应式编程的核心类型之一：</p>
     * <ul>
     *   <li>Mono&lt;T&gt;：0..1 个元素的异步结果</li>
     *   <li>Flux&lt;T&gt;：0..N 个元素的异步流</li>
     * </ul>
     *
     * @param message 用户输入
     * @return 逐 Token 输出的 Flux 流
     */
    public Flux<String> streamChat(String message) {
        log.debug("[流式对话] 用户消息: {}", message);

        return chatClient.prompt()
                .user(message)
                .stream()           // ← 关键区别：stream() 而非 call()
                .content();         // 返回 Flux<String>，每个元素是一个 Token
    }
}
