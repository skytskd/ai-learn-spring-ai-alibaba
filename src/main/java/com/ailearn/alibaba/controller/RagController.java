package com.ailearn.alibaba.controller;

import com.ailearn.alibaba.model.ChatRequest;
import com.ailearn.alibaba.service.RagService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * <h1>RagController — RAG 检索增强生成接口</h1>
 *
 * <p>提供知识库问答能力，AI 回答基于本地文档而非仅靠模型记忆。</p>
 *
 * <h2>对比实验</h2>
 * <pre>
 * // 无 RAG：模型不知道"阿里云百炼平台"
 * POST /api/chat/client
 * { "message": "阿里云百炼平台的计费方式是什么？" }
 * → 模型可能"编造"或说不知道
 *
 * // 有 RAG：基于本地文档回答
 * POST /api/rag/ask
 * { "message": "阿里云百炼平台的计费方式是什么？" }
 * → 基于 rag-docs/ 中的文档准确回答
 * </pre>
 *
 * @author ai-learn
 */
@Slf4j
@RestController
@RequestMapping("/api/rag")
@RequiredArgsConstructor
public class RagController {

    private final RagService ragService;

    /**
     * RAG 知识库问答（使用 QuestionAnswerAdvisor）
     */
    @PostMapping("/ask")
    public Map<String, String> ask(@RequestBody ChatRequest request) {
        log.info("[RAG问答] {}", request.getMessage());
        String answer = ragService.ask(request.getMessage());
        return Map.of("content", answer);
    }

    /**
     * RAG 手动实现版本（展示完整流程）
     */
    @PostMapping("/ask-manual")
    public Map<String, String> askManual(@RequestBody ChatRequest request) {
        log.info("[RAG手动] {}", request.getMessage());
        String answer = ragService.askWithManualRag(request.getMessage());
        return Map.of("content", answer);
    }
}
