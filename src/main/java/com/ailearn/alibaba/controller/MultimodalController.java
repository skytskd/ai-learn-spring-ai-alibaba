package com.ailearn.alibaba.controller;

import com.ailearn.alibaba.service.MultimodalService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.Resource;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

/**
 * <h1>MultimodalController — 多模态接口</h1>
 *
 * <p>支持上传图片并让 AI 分析图片内容。</p>
 *
 * <h2>使用方式</h2>
 * <pre>
 * curl -X POST http://localhost:8080/api/multimodal/analyze \
 *   -F "image=@photo.jpg" \
 *   -F "question=描述这张图片"
 * </pre>
 *
 * @author ai-learn
 */
@Slf4j
@RestController
@RequestMapping("/api/multimodal")
@RequiredArgsConstructor
public class MultimodalController {

    private final MultimodalService multimodalService;

    /**
     * 图片分析接口
     *
     * @param image    上传的图片文件
     * @param question 关于图片的问题（可选）
     * @return AI 的分析结果
     */
    @PostMapping(value = "/analyze", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Map<String, String> analyzeImage(
            @RequestParam("image") MultipartFile image,
            @RequestParam(value = "question", required = false) String question
    ) {
        log.info("[多模态] 图片分析: {}, 问题: {}", image.getOriginalFilename(), question);

        // MultipartFile → Resource
        Resource imageResource = image.getResource();

        String result = multimodalService.analyzeImage(imageResource, question);
        return Map.of("content", result);
    }
}
