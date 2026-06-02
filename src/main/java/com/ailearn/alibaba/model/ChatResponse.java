package com.ailearn.alibaba.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * <h1>ChatResponse — 聊天响应模型</h1>
 *
 * <p>后端返回给前端的聊天响应数据结构。</p>
 *
 * @author ai-learn
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ChatResponse {

    /** 对话ID，多轮对话中保持一致 */
    private String conversationId;

    /** AI 的回复内容 */
    private String content;

    /** 请求耗时（毫秒），用于性能对比 */
    private long elapsedMs;
}
