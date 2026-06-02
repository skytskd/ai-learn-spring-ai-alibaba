package com.ailearn.alibaba.controller;

import com.ailearn.alibaba.service.StructuredOutputService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * <h1>StructuredOutputController — 结构化输出接口</h1>
 *
 * <p>让 LLM 输出结构化的 JSON 数据，而非自由文本。
 * 这是 AI 应用对接业务系统的关键能力。</p>
 *
 * @author ai-learn
 */
@Slf4j
@RestController
@RequestMapping("/api/structured")
@RequiredArgsConstructor
public class StructuredOutputController {

    private final StructuredOutputService structuredOutputService;

    /**
     * 博客文章结构化提取
     * LLM 以指定格式输出标题、摘要、标签
     */
    @PostMapping("/blog-post")
    public BlogPost generateBlogPost(@RequestBody Map<String, String> request) {
        String topic = request.getOrDefault("topic", "Spring AI Alibaba");
        log.info("[结构化输出] 博客文章: {}", topic);

        String prompt = String.format("""
                请围绕"%s"写一篇博客文章的元数据。请以JSON格式返回，
                包含以下字段：title（标题）、summary（摘要）、tags（标签列表，3-5个）。
                """, topic);

        return structuredOutputService.outputAsBean(prompt, BlogPost.class);
    }

    /**
     * 博客文章 Bean（内嵌定义，演示结构化输出）
     */
    public record BlogPost(
            String title,
            String summary,
            List<String> tags
    ) {}
}
