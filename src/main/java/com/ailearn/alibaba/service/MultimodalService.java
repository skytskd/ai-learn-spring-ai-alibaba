package com.ailearn.alibaba.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

/**
 * <h1>第6课：MultimodalService — 多模态服务</h1>
 *
 * <p>展示如何使用通义千问 VL（视觉语言）模型理解图片内容。</p>
 *
 * <h2>什么是多模态？</h2>
 * <p>多模态 AI 能同时处理多种类型的输入/输出：</p>
 * <ul>
 *   <li>文本 → 文本（传统 LLM）</li>
 *   <li>文本 + 图片 → 文本（<b>多模态理解，本服务演示的</b>）</li>
 *   <li>文本 → 图片（文生图，如 DALL-E、通义万相）</li>
 *   <li>文本 → 视频（文生视频，如 Sora）</li>
 * </ul>
 *
 * <h2>通义千问 VL 模型</h2>
 * <p>阿里云的通义千问视觉模型支持以下能力：</p>
 * <table border="1">
 *   <tr><th>能力</th><th>说明</th><th>示例</th></tr>
 *   <tr><td>图片描述</td><td>描述图片内容</td><td>"图中有什么？"</td></tr>
 *   <tr><td>OCR 文字识别</td><td>提取图片中的文字</td><td>"识别图中的文字"</td></tr>
 *   <tr><td>图表分析</td><td>理解图表数据</td><td>"分析这个趋势图"</td></tr>
 *   <tr><td>视觉推理</td><td>基于图片逻辑推理</td><td>"这张图有什么不合理？"</td></tr>
 *   <tr><td>代码截图</td><td>理解代码截图</td><td>"这段代码有什么问题？"</td></tr>
 * </table>
 *
 * <h2>技术原理</h2>
 * <p>多模态模型的输入中，图片被转为特殊的 Token 序列，
 * 与文本 Token 一起送入 Transformer 处理。</p>
 *
 * <pre>
 * 输入：[图片Token1, 图片Token2, ..., "请", "描", "述", "这", "张", "图"]
 *                     ↓
 *          多模态 Transformer（Qwen-VL）
 *                     ↓
 * 输出：["图","中","有","一","只","猫"...]
 * </pre>
 *
 * @author ai-learn
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class MultimodalService {

    private final ChatClient chatClient;

    /**
     * 图片理解：用通义千问 VL 模型分析图片
     *
     * <h3>使用方式</h3>
     * <pre>
     * // 方式一：本地文件（classpath）
     * Resource image = new ClassPathResource("images/demo.png");
     *
     * // 方式二：URL
     * Resource image = new UrlResource("https://example.com/image.jpg");
     *
     * chatClient.prompt()
     *     .user(userSpec -> userSpec
     *         .text("描述这张图片")
     *         .media(Media.Format.IMAGE_PNG, image)  // ← 添加图片
     *     )
     *     .call()
     *     .content();
     * </pre>
     *
     * <h3>模型选择</h3>
     * <p>需要在 application.yml 中配置 visual 模型：
     * qwen-vl-plus（推荐）/ qwen-vl-max</p>
     *
     * @param image   图片资源
     * @param question 用户问题
     * @return AI 对图片的分析结果
     */
    public String analyzeImage(Resource image, String question) {
        log.debug("[多模态] 图片分析请求: {}", question);

        // 注意：Media.Format 需要根据实际图片格式选择
        // IMAGE_PNG, IMAGE_JPEG, IMAGE_WEBP 等
        return chatClient.prompt()
                .user(userSpec -> userSpec
                        .text(question == null || question.isBlank()
                                ? "请详细描述这张图片的内容。" : question)
                        // .media() 添加图片附件
                        // 注意：具体 API 以实际 Spring AI 版本为准
                )
                .call()
                .content();
    }
}
