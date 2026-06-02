package com.ailearn.alibaba.controller;

import com.ailearn.alibaba.service.StreamChatService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

/**
 * <h1>StreamChatController — 流式对话接口</h1>
 *
 * <p>通过 SSE（Server-Sent Events）实现 AI 回复的逐字输出。
 * 支持跨域，方便前端直接通过 EventSource 消费。</p>
 *
 * <h2>前端调用示例</h2>
 * <pre>
 * const eventSource = new EventSource(
 *     '/api/stream/chat?message=你好'
 * );
 *
 * eventSource.onmessage = (event) => {
 *     console.log(event.data);  // 每个 Token
 * };
 * </pre>
 *
 * @author ai-learn
 */
@Slf4j
@RestController
@RequestMapping("/api/stream")
@RequiredArgsConstructor
public class StreamChatController {

    private final StreamChatService streamChatService;

    /**
     * 流式对话接口
     *
     * <p>返回类型是 {@code Flux<String>}，Spring WebFlux 自动将其
     * 序列化为 SSE 格式（text/event-stream）。</p>
     *
     * <h3>对比普通接口</h3>
     * <table border="1">
     *   <tr><th></th><th>普通接口</th><th>流式接口</th></tr>
     *   <tr><td>Content-Type</td><td>application/json</td><td>text/event-stream</td></tr>
     *   <tr><td>返回方式</td><td>一次性返回完整 JSON</td><td>逐行返回 data: token</td></tr>
     *   <tr><td>用户体验</td><td>等待后全部出现</td><td>逐字打字效果</td></tr>
     *   <tr><td>实现方式</td><td>@RestController 返回 Bean</td><td>Flux + SSE</td></tr>
     * </table>
     */
    @GetMapping(value = "/chat", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> streamChat(@RequestParam String message) {
        log.info("[SSE流式] 用户消息: {}", message);
        return streamChatService.streamChat(message);
    }
}
