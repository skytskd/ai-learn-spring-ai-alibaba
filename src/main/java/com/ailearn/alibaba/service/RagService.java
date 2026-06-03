package com.ailearn.alibaba.service;

import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.rag.advisor.RetrievalAugmentationAdvisor;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * <h1>第4课：RagService — RAG 检索增强生成</h1>
 *
 * <p>展示完整的 RAG（Retrieval Augmented Generation）流程。</p>
 *
 * @author ai-learn
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class RagService {

    private final ChatClient chatClient;
    private final EmbeddingModel embeddingModel;
    private final ResourceLoader resourceLoader;

    /**
     * 向量存储（基于内存，学习用）
     */
    private VectorStore vectorStore;

    /**
     * 自定义 RAG 检索返回的最大文档数
     */
    @Value("${ai-learn.rag.top-k:4}")
    private int topK;

    // ==================== 初始化：构建知识库 ====================

    /**
     * 应用启动时，加载文档并构建向量索引
     */
    @PostConstruct
    public void init() {
        log.info("[RAG] 开始构建知识库索引...");

        try {
            List<Document> allDocuments = new ArrayList<>();

            // 加载 resources/rag-docs/ 下的所有 txt 文件
            Resource[] resources = {
                    resourceLoader.getResource("classpath:/rag-docs/spring-ai-intro.txt"),
                    resourceLoader.getResource("classpath:/rag-docs/alibaba-cloud-intro.txt")
            };

            for (Resource resource : resources) {
                if (resource.exists()) {
                    TextReader reader = new TextReader(resource);
                    List<Document> docs = reader.get();
                    log.info("[RAG] 加载文档: {}, 内容长度: {} 字符",
                            resource.getFilename(),
                            docs.stream().mapToInt(d -> d.getText().length()).sum());

                    allDocuments.addAll(docs);
                }
            }

            if (!allDocuments.isEmpty()) {
                vectorStore = SimpleVectorStore.builder(embeddingModel).build();
                vectorStore.add(allDocuments);
                log.info("[RAG] 知识库索引构建完成，共 {} 篇文档", allDocuments.size());
            } else {
                log.warn("[RAG] 未找到任何文档，RAG 功能不可用");
            }

        } catch (Exception e) {
            log.error("[RAG] 知识库索引构建失败", e);
        }
    }

    // ==================== RAG 问答 ====================

    /**
     * 基于 RAG 的知识库问答（方式一：使用 RetrievalAugmentationAdvisor）
     *
     * <p>Spring AI 1.1.0 中 QuestionAnswerAdvisor 已被移除，
     * 替换为 RetrievalAugmentationAdvisor（位于 spring-ai-rag 模块）。
     * 需要配合 VectorStoreDocumentRetriever 使用。</p>
     *
     * @param question 用户问题
     * @return 基于知识库的回答
     */
    public String ask(String question) {
        if (vectorStore == null) {
            return "⚠️ 知识库未初始化，请确保 rag-docs 目录下有文档文件。";
        }

        log.debug("[RAG问答] 问题: {}", question);

        // Spring AI 1.1.0：使用 RetrievalAugmentationAdvisor 替代 QuestionAnswerAdvisor
        VectorStoreDocumentRetriever retriever = VectorStoreDocumentRetriever.builder()
                .vectorStore(vectorStore)
                .topK(topK)
                .build();

        RetrievalAugmentationAdvisor advisor = RetrievalAugmentationAdvisor.builder()
                .documentRetriever(retriever)
                .build();

        return chatClient.prompt()
                .user(question)
                .advisors(advisor)
                .call()
                .content();
    }

    /**
     * 手动实现 RAG（方式二）：展示完整流程
     *
     * <p>Spring AI 1.1.0 中 VectorStore.similaritySearch(String, int) 已废弃，
     * 需要使用 SearchRequest.builder() 构建检索请求。</p>
     *
     * @param question 用户问题
     * @return 带来源引用的回答
     */
    public String askWithManualRag(String question) {
        if (vectorStore == null) {
            return "⚠️ 知识库未初始化。";
        }

        log.debug("[RAG手动] 问题: {}", question);

        // Spring AI 1.1.0：使用 SearchRequest 替代旧的双参数方法
        List<Document> relevantDocs = vectorStore.similaritySearch(
                SearchRequest.builder()
                        .query(question)
                        .topK(topK)
                        .build()
        );

        if (relevantDocs.isEmpty()) {
            return "未找到相关知识，请尝试换一种问法。";
        }

        log.debug("[RAG手动] 检索到 {} 篇相关文档", relevantDocs.size());

        // 构建上下文
        StringBuilder context = new StringBuilder();
        for (int i = 0; i < relevantDocs.size(); i++) {
            context.append("【参考资料")
                    .append(i + 1)
                    .append("】\n")
                    .append(relevantDocs.get(i).getText())
                    .append("\n\n");
        }

        // 拼接 Prompt 并调用模型
        String prompt = """
                请根据以下参考资料回答用户问题。
                如果参考资料中没有相关信息，请如实告知，不要编造。

                参考资料：
                %s

                用户问题：%s

                请用中文回答，并在回答末尾标注引用的资料编号。
                """.formatted(context.toString(), question);

        return chatClient.prompt()
                .user(prompt)
                .call()
                .content();
    }
}
