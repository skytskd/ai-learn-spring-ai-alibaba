package com.ailearn.alibaba.service;

import com.ailearn.alibaba.model.ChatResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.stereotype.Service;

import java.util.Map;

/**
 * <h1>第1课：BasicChatService — 基础对话服务</h1>
 *
 * <p>展示 Spring AI 中三种核心的对话调用方式，从底层到高层，
 * 理解 AI 调用的完整链路。</p>
 *
 * <h2>三种调用方式对比</h2>
 * <table border="1">
 *   <tr><th>方式</th><th>API</th><th>灵活度</th><th>适用场景</th></tr>
 *   <tr><td>方式一</td><td>ChatModel.call()</td><td>⭐⭐⭐ 最高</td><td>底层定制、复杂 Prompt 组装</td></tr>
 *   <tr><td>方式二</td><td>PromptTemplate</td><td>⭐⭐ 中等</td><td>参数化 Prompt、模板复用</td></tr>
 *   <tr><td>方式三</td><td>ChatClient.call()</td><td>⭐ 最简</td><td>日常开发、快速接入（推荐）</td></tr>
 * </table>
 *
 * <h2>核心概念</h2>
 * <ul>
 *   <li><b>ChatModel</b>：底层模型接口，直接调用 AI API</li>
 *   <li><b>Prompt</b>：一次请求的完整上下文（System Message + User Message + History）</li>
 *   <li><b>PromptTemplate</b>：支持占位符的模板，运行时填充变量</li>
 *   <li><b>ChatClient</b>：高层封装，Builder 模式链式调用</li>
 * </ul>
 *
 * @author ai-learn
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class BasicChatService {

    private final ChatModel chatModel;       // 底层模型接口（直接调用）
    private final ChatClient chatClient;     // 高层封装（推荐使用）

    // ==================== 方式一：ChatModel.call() ====================

    /**
     * <h3>方式一：ChatModel 直接调用</h3>
     *
     * <p>最底层的调用方式，需要手动构建 {@link Prompt} 对象。
     * 所有高层 API 最终都会转为这种方式。</p>
     *
     * <h3>执行流程</h3>
     * <ol>
     *   <li>构建 {@link UserMessage}（用户消息）</li>
     *   <li>构建 {@link Prompt}（完整的对话上下文）</li>
     *   <li>调用 {@code chatModel.call(prompt)} 获取响应</li>
     *   <li>从 {@code ChatResponse.getResult().getOutput()} 提取文本</li>
     * </ol>
     *
     * <h3>适用场景</h3>
     * <ul>
     *   <li>需要精细控制 Prompt 结构</li>
     *   <li>批量处理（一次发多条消息）</li>
     *   <li>自定义消息类型（如 Function Message）</li>
     * </ul>
     *
     * @param message 用户输入
     * @return AI 回复内容
     */
    public ChatResponse chatWithModel(String message) {
        log.debug("[方式一] ChatModel.call() — 用户消息: {}", message);

        long start = System.currentTimeMillis();

        // 1️⃣ 创建 UserMessage
        // Message 是 Spring AI 中的消息抽象，有多种类型：
        //   - UserMessage：用户消息
        //   - SystemMessage：系统提示词
        //   - AssistantMessage：AI 回复历史
        //   - FunctionMessage：函数调用结果
        UserMessage userMessage = new UserMessage(message);

        // 2️⃣ 构建 Prompt
        // Prompt 包含一次请求的全部上下文
        Prompt prompt = new Prompt(userMessage);

        // 3️⃣ 调用模型
        // chatModel.call() 是阻塞调用，返回 ChatResponse（Spring AI 的响应对象）
        org.springframework.ai.chat.model.ChatResponse response = chatModel.call(prompt);

        // 4️⃣ 提取文本内容
        // getResult() — 获取第一个生成结果
        // getOutput().getText() — 提取文本
        String content = response.getResult().getOutput().getText();

        long elapsed = System.currentTimeMillis() - start;
        log.debug("[方式一] 响应耗时: {}ms, 内容长度: {}", elapsed, content.length());

        return ChatResponse.builder()
                .content(content)
                .elapsedMs(elapsed)
                .build();
    }

    // ==================== 方式二：PromptTemplate ====================

    /**
     * <h3>方式二：PromptTemplate 模板调用</h3>
     *
     * <p>适合需要参数化 Prompt 的场景，例如同一个提示词模板
     * 填入不同的变量值。</p>
     *
     * <h3>PromptTemplate 语法</h3>
     * <pre>
     * // 在模板中使用 {key} 占位符
     * String template = "请用 {language} 解释 {concept}";
     *
     * // 运行时填充
     * PromptTemplate pt = new PromptTemplate(template);
     * pt.add("language", "Java");
     * pt.add("concept", "依赖注入");
     * </pre>
     *
     * <h3>与方式一的区别</h3>
     * <ul>
     *   <li>方式一：手动拼装 Message 对象列表</li>
     *   <li>方式二：通过模板引擎自动渲染，支持变量替换</li>
     * </ul>
     *
     * @param topic 要学习的技术主题
     * @return AI 的教学内容
     */
    public ChatResponse chatWithTemplate(String topic) {
        log.debug("[方式二] PromptTemplate — 主题: {}", topic);

        long start = System.currentTimeMillis();

        // 1️⃣ 定义模板
        // {topic} 是占位符，运行时替换
        String template = """
                你是一个 Java 技术专家。请用通俗易懂的方式解释以下概念：

                概念：{topic}

                要求：
                1. 先给出一句话定义
                2. 用一个生活中的类比帮助理解
                3. 给一个简单的代码示例
                4. 说明使用场景和注意事项
                """;

        // 2️⃣ 创建 PromptTemplate 并填充变量
        PromptTemplate promptTemplate = new PromptTemplate(template);
        promptTemplate.add("topic", topic);

        // 3️⃣ 创建 Prompt 并调用
        // promptTemplate.create() 将模板渲染为 Prompt 对象
        org.springframework.ai.chat.model.ChatResponse response =
                chatModel.call(promptTemplate.create());

        String content = response.getResult().getOutput().getText();
        long elapsed = System.currentTimeMillis() - start;

        return ChatResponse.builder()
                .content(content)
                .elapsedMs(elapsed)
                .build();
    }

    // ==================== 方式三：ChatClient（推荐）====================

    /**
     * <h3>方式三：ChatClient 链式调用（推荐）</h3>
     *
     * <p>{@link ChatClient} 是 Spring AI 对 ChatModel 的高层封装，
     * 提供了流畅的 Builder API，是日常开发的首选方式。</p>
     *
     * <h3>核心方法链</h3>
     * <pre>
     * chatClient.prompt()      // 开始构建请求
     *     .user("message")     // 设置用户消息
     *     .system("prompt")    // 覆盖默认 System Prompt
     *     .advisors(...)       // 添加 Advisor（日志/记忆/安全等）
     *     .call()              // 发送请求，阻塞等待
     *     .content();          // 提取文本内容
     * </pre>
     *
     * <h3>为什么推荐 ChatClient？</h3>
     * <ul>
     *   <li>链式调用，代码可读性高</li>
     *   <li>内置 Advisor 链机制（AOP 风格）</li>
     *   <li>自动管理 System Prompt</li>
     *   <li>统一的流式和非流式接口</li>
     * </ul>
     *
     * @param message 用户输入
     * @return AI 回复内容
     */
    public ChatResponse chatWithClient(String message) {
        log.debug("[方式三] ChatClient.call() — 用户消息: {}", message);

        long start = System.currentTimeMillis();

        // 一行链式调用完成对话
        // .user() 设置用户消息
        // .call() 发送请求并阻塞等待
        // .content() 直接提取文本（无需手动处理 ChatResponse）
        String content = chatClient.prompt()
                .user(message)
                .call()
                .content();

        long elapsed = System.currentTimeMillis() - start;

        return ChatResponse.builder()
                .content(content)
                .elapsedMs(elapsed)
                .build();
    }
}
