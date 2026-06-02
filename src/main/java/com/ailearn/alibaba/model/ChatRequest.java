package com.ailearn.alibaba.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * <h1>ChatRequest — 聊天请求模型</h1>
 *
 * <p>前端发送的聊天请求数据结构。</p>
 *
 * <h2>REST API 示例</h2>
 * <pre>
 * POST /api/chat/simple
 * Content-Type: application/json
 *
 * {
 *   "message": "什么是 Spring AI Alibaba？",
 *   "conversationId": "conv-001"
 * }
 * </pre>
 *
 * @author ai-learn
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class ChatRequest {

    /**
     * 用户输入的消息内容
     */
    private String message;

    /**
     * 对话ID，用于多轮对话的记忆关联
     * 为空时系统自动生成
     */
    private String conversationId;
}
