package com.ailearn.alibaba.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

/**
 * <h1>第7课：StructuredOutputService — 结构化输出服务</h1>
 *
 * <p>展示如何让 LLM 输出结构化的数据（JSON → Java Bean），
 * 而非自由文本。这是 AI 应用对接业务系统的关键能力。</p>
 *
 * <h2>为什么需要结构化输出？</h2>
 *
 * <pre>
 * ❌ 自由文本（难以程序化处理）：
 * "今天的天气是晴天，温度25度，湿度60%"
 *
 * ✅ 结构化输出（直接映射为 Java Bean）：
 * {
 *   "city": "杭州",
 *   "condition": "晴",
 *   "temperature": 25.0,
 *   "humidity": 60
 * }
 * </pre>
 *
 * <h2>实现方式</h2>
 *
 * <h3>entity() 方法</h3>
 * <p>Spring AI 的 ChatClient 提供了 {@code .entity()} 方法，
 * 会自动将 LLM 返回的内容解析为指定的 Java 类型。</p>
 *
 * <pre>
 * WeatherInfo result = chatClient.prompt()
 *     .user("杭州今天天气怎么样？")
 *     .call()
 *     .entity(WeatherInfo.class);  // 自动 JSON → Bean
 * </pre>
 *
 * <h3>工作流程</h3>
 * <ol>
 *   <li>在 Prompt 中要求 LLM 输出 JSON 格式</li>
 *   <li>LLM 返回 JSON 字符串</li>
 *   <li>Spring AI 使用 Jackson 自动反序列化为 Java Bean</li>
 * </ol>
 *
 * <h2>应用场景</h2>
 * <ul>
 *   <li>信息提取：从文本中提取结构化信息</li>
 *   <li>智能表单：自然语言 → 表单数据</li>
 *   <li>数据转换：非结构化数据 → 结构化数据</li>
 *   <li>API 编排：LLM 输出作为下游 API 的输入</li>
 * </ul>
 *
 * @author ai-learn
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class StructuredOutputService {

    private final ChatClient chatClient;

    /**
     * 结构化输出：将 LLM 回复映射为 Java Bean
     *
     * <h3>使用示例</h3>
     * <pre>
     * // 假设有一个 BlogPost 类：
     * record BlogPost(String title, String summary, List&lt;String&gt; tags) {}
     *
     * BlogPost post = service.generateStructuredOutput(
     *     "请写一篇关于 Spring AI 的博客文章",
     *     BlogPost.class
     * );
     * </pre>
     *
     * <h3>注意事项</h3>
     * <ul>
     *   <li>LLM 并非每次都严格遵循 JSON 格式，可能需要重试</li>
     *   <li>对于关键业务场景，建议添加格式校验和异常处理</li>
     *   <li>复杂嵌套结构可能增加输出错误率</li>
     * </ul>
     *
     * @param <T>       目标 Java 类型
     * @param prompt    给 LLM 的提示词
     * @param beanClass 目标 Bean 的 Class 对象
     * @return 解析后的 Bean 实例
     */
    public <T> T outputAsBean(String prompt, Class<T> beanClass) {
        log.debug("[结构化输出] 类型: {}, prompt: {}", beanClass.getSimpleName(), prompt);

        return chatClient.prompt()
                .user(prompt)
                .call()
                // .entity() 是结构化输出的关键
                // 它会要求 LLM 输出 JSON 格式，并自动反序列化
                .entity(beanClass);
    }
}
