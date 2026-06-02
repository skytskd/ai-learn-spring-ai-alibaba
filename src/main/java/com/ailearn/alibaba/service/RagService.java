package com.ailearn.alibaba.service;

import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.QuestionAnswerAdvisor;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.TextReader;
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
 * <p>展示完整的 RAG（Retrieval Augmented Generation）流程，
 * 是当前 AI 应用最核心的架构模式之一。</p>
 *
 * <h2>什么是 RAG？</h2>
 * <p>RAG = 检索（Retrieval）+ 增强（Augmented）+ 生成（Generation）</p>
 *
 * <pre>
 *                     RAG 工作流程
 *  ┌──────────────────────────────────────────────────────┐
 *  │                    离线阶段（索引构建）                  │
 *  │                                                      │
 *  │  文档 ──→ 加载 ──→ 分块 ──→ Embedding ──→ 向量数据库  │
 *  └──────────────────────────────────────────────────────┘
 *                            │
 *                            ▼
 *  ┌──────────────────────────────────────────────────────┐
 *  │                    在线阶段（问答检索）                  │
 *  │                                                      │
 *  │  用户问题 ──→ Embedding ──→ 向量检索 ──→ 相关文档      │
 *  │                                              │       │
 *  │                                              ▼       │
 *  │              问题 + 相关文档 ──→ LLM ──→ 回答         │
 *  └──────────────────────────────────────────────────────┘
 * </pre>
 *
 * <h2>为什么需要 RAG？</h2>
 * <ul>
 *   <li><b>知识截止</b>：大模型训练数据有截止日期，RAG 可注入最新知识</li>
 *   <li><b>私有知识</b>：企业文档、内部资料无法训练进模型，RAG 可检索使用</li>
 *   <li><b>可追溯</b>：回答可以引用源文档，提高可信度</li>
 *   <li><b>幻觉控制</b>：限定在检索到的文档范围内回答，减少编造</li>
 * </ul>
 *
 * <h2>核心概念</h2>
 * <ul>
 *   <li><b>Document</b>：文档对象，包含文本内容和元数据</li>
 *   <li><b>TextReader</b>：文档加载器，从文件读取文本</li>
 *   <li><b>TokenTextSplitter</b>：文本分块器，按 Token 数切分长文档</li>
 *   <li><b>EmbeddingModel</b>：向量化模型，将文本转为向量</li>
 *   <li><b>VectorStore</b>：向量存储，支持相似度检索</li>
 *   <li><b>QuestionAnswerAdvisor</b>：Spring AI 内置的 RAG Advisor</li>
 * </ul>
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
     * 生产环境建议使用 Elasticsearch、Milvus、PGVector 等
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
     *
     * <h3>索引构建步骤</h3>
     * <ol>
     *   <li><b>加载文档</b>：从 classpath 读取 txt 文件</li>
     *   <li><b>文本分块</b>：将长文档切分成语义完整的段落</li>
     *   <li><b>向量化</b>：调用 Embedding 模型将文本转为数值向量</li>
     *   <li><b>存储</b>：向量 + 原文本存入 VectorStore</li>
     * </ol>
     *
     * <h3>为什么需要分块？</h3>
     * <p>Embedding 模型有最大 Token 限制（通常 512~8192）。
     * 超过限制的文本需要切分。同时，较小的块能提高检索精度。</p>
     *
     * <pre>
     * 原始文档（10000 Token）
     * ┌─────────────────────────────────────┐
     * │ Chunk 1 (500 Token)                 │ ──→ Vector 1
     * │ Chunk 2 (500 Token)                 │ ──→ Vector 2
     * │ Chunk 3 (500 Token)                 │ ──→ Vector 3
     * │ ...                                │
     * └─────────────────────────────────────┘
     * </pre>
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
                    // TextReader：Spring AI 内置的文本文件加载器
                    TextReader reader = new TextReader(resource);
                    List<Document> docs = reader.get();
                    log.info("[RAG] 加载文档: {}, 内容长度: {} 字符",
                            resource.getFilename(),
                            docs.stream().mapToInt(d -> d.getText().length()).sum());

                    allDocuments.addAll(docs);
                }
            }

            if (!allDocuments.isEmpty()) {
                // 创建 SimpleVectorStore 并写入文档
                // SimpleVectorStore 会在添加文档时自动完成：
                //   1. 调用 EmbeddingModel 向量化
                //   2. 构建内存索引
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
     * 基于 RAG 的知识库问答
     *
     * <h3>两种实现方式</h3>
     *
     * <h4>方式一：QuestionAnswerAdvisor（推荐，简单）</h4>
     * <p>Spring AI 内置的 RAG Advisor，一行代码完成检索增强：</p>
     * <pre>
     * chatClient.prompt()
     *     .advisors(new QuestionAnswerAdvisor(vectorStore))
     *     .call()
     *     .content();
     * </pre>
     *
     * <h4>方式二：手动实现（灵活，可定制）</h4>
     * <p>自己控制检索、拼接 Prompt 的每个环节：</p>
     * <ol>
     *   <li>将用户问题向量化</li>
     *   <li>在 VectorStore 中检索 Top-K 相似文档</li>
     *   <li>将检索到的文档拼接到 System Prompt</li>
     *   <li>调用模型生成回答</li>
     * </ol>
     *
     * @param question 用户问题
     * @return 基于知识库的回答
     */
    public String ask(String question) {
        if (vectorStore == null) {
            return "⚠️ 知识库未初始化，请确保 rag-docs 目录下有文档文件。";
        }

        log.debug("[RAG问答] 问题: {}", question);

        // ========== 方式一：使用 QuestionAnswerAdvisor ==========
        // QuestionAnswerAdvisor 自动完成：
        //   1. 将用户问题向量化
        //   2. 在 VectorStore 中检索 Top-K 相似文档
        //   3. 将检索结果注入到 Prompt
        //   4. 让模型基于这些文档回答
        return chatClient.prompt()
                .user(question)
                .advisors(new QuestionAnswerAdvisor(vectorStore))
                .call()
                .content();
    }

    /**
     * 手动实现 RAG（方式二）：展示完整流程
     *
     * <p>与 QuestionAnswerAdvisor 不同，这里展示了
     * RAG 的每一步细节，方便理解底层原理。</p>
     *
     * @param question 用户问题
     * @return 带来源引用的回答
     */
    public String askWithManualRag(String question) {
        if (vectorStore == null) {
            return "⚠️ 知识库未初始化。";
        }

        log.debug("[RAG手动] 问题: {}", question);

        // 1️⃣ 向量检索
        // similaritySearch() 将问题向量化并检索最相似的 Top-K 文档
        // 参数：查询文本、最大返回数
        List<Document> relevantDocs = vectorStore.similaritySearch(
                question, topK
        );

        if (relevantDocs.isEmpty()) {
            return "未找到相关知识，请尝试换一种问法。";
        }

        log.debug("[RAG手动] 检索到 {} 篇相关文档", relevantDocs.size());

        // 2️⃣ 构建上下文
        StringBuilder context = new StringBuilder();
        for (int i = 0; i < relevantDocs.size(); i++) {
            context.append("【参考资料")
                    .append(i + 1)
                    .append("】\n")
                    .append(relevantDocs.get(i).getText())
                    .append("\n\n");
        }

        // 3️⃣ 拼接 Prompt 并调用模型
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
