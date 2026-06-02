package com.ailearn.alibaba.controller;

import com.ailearn.alibaba.model.ChatRequest;
import com.ailearn.alibaba.model.ChatResponse;
import com.ailearn.alibaba.service.BasicChatService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

/**
 * <h1>BasicChatController — 基础对话接口</h1>
 *
 * <p>提供三种不同抽象层次的对话接口，方便对比学习。</p>
 *
 * <h2>接口列表</h2>
 * <table border="1">
 *   <tr><th>接口</th><th>调用方式</th><th>特点</th></tr>
 *   <tr><td>POST /api/chat/model</td><td>ChatModel.call()</td><td>底层，灵活</td></tr>
 *   <tr><td>POST /api/chat/template</td><td>PromptTemplate</td><td>参数化模板</td></tr>
 *   <tr><td>POST /api/chat/client</td><td>ChatClient.call()</td><td>高层，简洁（推荐）</td></tr>
 * </table>
 *
 * @author ai-learn
 */
@Slf4j
@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
public class BasicChatController {

    private final BasicChatService basicChatService;

    /**
     * 方式一：ChatModel 直接调用
     */
    @PostMapping("/model")
    public ChatResponse chatWithModel(@RequestBody ChatRequest request) {
        log.info("[接口] ChatModel 方式: {}", request.getMessage());
        return basicChatService.chatWithModel(request.getMessage());
    }

    /**
     * 方式二：PromptTemplate 模板调用
     */
    @PostMapping("/template")
    public ChatResponse chatWithTemplate(@RequestBody ChatRequest request) {
        log.info("[接口] PromptTemplate 方式: {}", request.getMessage());
        return basicChatService.chatWithTemplate(request.getMessage());
    }

    /**
     * 方式三：ChatClient 链式调用（推荐）
     */
    @PostMapping("/client")
    public ChatResponse chatWithClient(@RequestBody ChatRequest request) {
        log.info("[接口] ChatClient 方式: {}", request.getMessage());
        return basicChatService.chatWithClient(request.getMessage());
    }
}
