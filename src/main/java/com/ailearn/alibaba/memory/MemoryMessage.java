package com.ailearn.alibaba.memory;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * <h1>MemoryMessage — 记忆中的单条消息</h1>
 *
 * <p>相比直接用 Spring AI 的 {@code Message} 抽象，本类额外携带了
 * 角色与时间戳，便于做摘要生成、事实抽取与调试展示。</p>
 *
 * <p>之所以不直接复用 {@code UserMessage} / {@code AssistantMessage}：
 * 那些类型被设计为「一次性请求上下文」，不适合作为需要序列化、
 * 持久化、可回溯的长期记忆载体。</p>
 *
 * @author ai-learn
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class MemoryMessage {

    /** 消息角色 */
    public enum Role {
        USER,
        ASSISTANT,
        /** 系统注入的摘要或事实，作为额外上下文参与拼接 */
        SYSTEM
    }

    private Role role;

    private String content;

    /** 消息产生时间（epoch 毫秒），便于排序与调试 */
    private long timestamp = System.currentTimeMillis();

    public MemoryMessage(Role role, String content) {
        this.role = role;
        this.content = content;
        this.timestamp = System.currentTimeMillis();
    }

    public static MemoryMessage user(String content) {
        return new MemoryMessage(Role.USER, content);
    }

    public static MemoryMessage assistant(String content) {
        return new MemoryMessage(Role.ASSISTANT, content);
    }

    public static MemoryMessage system(String content) {
        return new MemoryMessage(Role.SYSTEM, content);
    }

    /**
     * 转换为用于摘要 prompt 的文本行。
     */
    public String toPromptLine() {
        String speaker = switch (role) {
            case USER -> "用户";
            case ASSISTANT -> "助手";
            case SYSTEM -> "系统";
        };
        return speaker + ": " + content;
    }
}
