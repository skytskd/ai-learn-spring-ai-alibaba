package com.ailearn.alibaba;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * <h1>AiLearnAlibabaApplication — 启动类</h1>
 *
 * <p>基于 Spring AI Alibaba（阿里云通义千问）的 AI 应用进阶学习项目。
 * 本项目覆盖从基础对话到 Agent 工具调用的完整学习路径。</p>
 *
 * <h2>学习路线</h2>
 * <ol>
 *   <li>基础对话：ChatClient 的 3 种调用方式</li>
 *   <li>流式输出：Flux + SSE 实时响应</li>
 *   <li>对话记忆：InMemoryChatMemory 上下文保持</li>
 *   <li>RAG 检索增强：文档加载 → 分块 → 向量化 → 检索</li>
 *   <li>Agent 工具调用：@Tool 注解 + Function Calling</li>
 *   <li>多模态交互：图片理解（通义千问 VL）</li>
 *   <li>结构化输出：Bean 映射输出</li>
 * </ol>
 *
 * <h2>快速开始</h2>
 * <pre>
 * // 1. 配置 API Key（编辑 application.yml 或设置环境变量）
 * export DASHSCOPE_API_KEY=your-api-key
 *
 * // 2. 启动项目
 * mvn spring-boot:run
 *
 * // 3. 打开学习指南
 * http://localhost:8080
 * </pre>
 *
 * @author ai-learn
 * @see <a href="https://java2ai.com">Spring AI Alibaba 官方文档</a>
 * @see <a href="https://bailian.console.aliyun.com/">阿里云百炼平台</a>
 */
@SpringBootApplication
public class AiLearnAlibabaApplication {

    public static void main(String[] args) {
        SpringApplication.run(AiLearnAlibabaApplication.class, args);
    }
}
