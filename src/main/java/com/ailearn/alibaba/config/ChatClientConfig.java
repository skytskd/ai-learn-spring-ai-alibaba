package com.ailearn.alibaba.config;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * <h1>ChatClientConfig — AI 客户端配置</h1>
 *
 * <p>将 Spring AI Alibaba 提供的 {@link ChatModel}（通义千问模型）封装为
 * 更易用的 {@link ChatClient}，提供流式调用、系统提示词、Advisor 链等功能。</p>
 *
 * <h2>ChatClient vs ChatModel</h2>
 * <table border="1">
 *   <tr><th>特性</th><th>ChatModel</th><th>ChatClient</th></tr>
 *   <tr><td>调用方式</td><td>底层 call() 方法</td><td>流式 Builder API</td></tr>
 *   <tr><td>System Prompt</td><td>需手动拼装 Message</td><td>.system() 直接设置</td></tr>
 *   <tr><td>Advisor 链</td><td>需手动传递</td><td>.advisors() 链式配置</td></tr>
 *   <tr><td>推荐场景</td><td>底层定制</td><td>日常开发（推荐）</td></tr>
 * </table>
 *
 * <h2>关键设计：原型 Bean</h2>
 * <p>ChatClient 本身是线程安全的，但通过
 * {@link ChatClient.Builder} 创建的实例会绑定静态 System Prompt。
 * 如果需要动态 System Prompt，每次调用时使用 builder 重新构建。</p>
 *
 * @author ai-learn
 */
@Configuration
public class ChatClientConfig {

    /**
     * 创建 ChatClient Bean，注入通义千问 ChatModel
     *
     * <p>Spring AI Alibaba 的 DashScope Starter 会自动创建
     * {@link ChatModel} Bean（DashScopeChatModel），
     * 这里将其包装为 ChatClient。</p>
     *
     * @param chatModel Spring AI Alibaba 自动配置的 ChatModel（DashScopeChatModel）
     * @return ChatClient 实例
     */
    @Bean
    public ChatClient chatClient(ChatModel chatModel) {
        // ChatClient.builder() 是创建入口
        // .defaultSystem() — 设置默认的系统提示词（所有请求都会带上）
        // .defaultAdvisors() — 设置默认的 Advisor 链（如日志记录、记忆管理等）
        return ChatClient.builder(chatModel)
                // 默认系统提示词：定义 AI 助手的角色和行为
                .defaultSystem("""
                        你是一个专业的 AI 编程学习助手，名叫"通义学伴"。
                        你的特点：
                        1. 使用中文回复，代码注释也用中文
                        2. 回答简洁清晰，重点突出
                        3. 当解释技术概念时，使用类比帮助理解
                        4. 代码示例直接可用，包含必要的依赖和配置
                        """)
                .build();
    }
}
