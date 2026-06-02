# 🧠 ai-learn-spring-ai-alibaba — AI 应用开发 Java 进阶学习项目

基于 **Spring Boot 3.4 + Spring AI Alibaba 1.1.0** 的 AI 应用开发教学项目，使用阿里云通义千问系列模型。

## 🎯 与 ai-learn（OpenAI 版）的区别

| 对比维度 | ai-learn (OpenAI) | ai-learn-spring-ai-alibaba |
|---------|-------------------|---------------------------|
| AI 模型 | OpenAI (GPT-4o 等) | 通义千问 (qwen-plus 等) |
| API 平台 | OpenAI API | 阿里云百炼 DashScope |
| 国内访问 | 需要代理 | 直接访问，低延迟 |
| 中文能力 | 较好 | 更强（原生中文优化） |
| 成本 | 较高 | 较低（有免费额度） |
| 多模态 | 单独 API | 通义 VL 原生支持 |

## 📚 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Spring Boot | 3.4.5 | 基础框架 |
| Spring AI | 1.1.0 | AI 标准接口层 |
| Spring AI Alibaba | 1.1.0.0 | 阿里云模型集成 |
| DashScope | — | 通义千问 API |
| Spring WebFlux | 3.4.5 | SSE 流式接口 |
| Lombok | latest | 减少样板代码 |

## 🗺️ 学习路线

| 课程 | 模块 | 核心知识点 | 接口 |
|------|------|-----------|------|
| 第 1 课 | BasicChatService | ChatModel、PromptTemplate、ChatClient 三种调用方式 | `POST /api/chat/client` |
| 第 2 课 | StreamChatService | Flux、SSE、响应式流 | `GET /api/stream/chat` |
| 第 3 课 | MemoryChatService | ChatMemory、MessageChatMemoryAdvisor | `POST /api/memory/chat` |
| 第 4 课 | RagService | 文档加载→分块→Embedding→向量检索 | `POST /api/rag/ask` |
| 第 5 课 | AgentService | @Tool、Function Calling、ReAct 模式 | `POST /api/agent/chat` |
| 第 6 课 | MultimodalService | 通义 VL 图片理解 | `POST /api/multimodal/analyze` |
| 第 7 课 | StructuredOutputService | Bean 映射、结构化 JSON 输出 | `POST /api/structured/blog-post` |

## 📁 项目结构

```
ai-learn-spring-ai-alibaba/
├── pom.xml                                    # Maven 配置
├── src/main/java/com/ailearn/alibaba/
│   ├── AiLearnAlibabaApplication.java         # 启动类
│   ├── config/
│   │   └── ChatClientConfig.java              # ChatClient Bean 配置
│   ├── controller/
│   │   ├── BasicChatController.java           # 基础对话接口（3种方式）
│   │   ├── StreamChatController.java          # SSE 流式接口
│   │   ├── MemoryChatController.java          # 记忆对话接口
│   │   ├── RagController.java                 # RAG 问答接口
│   │   ├── AgentController.java               # Agent 接口
│   │   ├── MultimodalController.java          # 多模态接口
│   │   └── StructuredOutputController.java    # 结构化输出接口
│   ├── service/
│   │   ├── BasicChatService.java              # 基础对话（详细注释）
│   │   ├── StreamChatService.java             # 流式输出
│   │   ├── MemoryChatService.java             # 对话记忆
│   │   ├── RagService.java                    # RAG 系统
│   │   ├── AgentService.java                  # Agent 工具调用
│   │   ├── MultimodalService.java             # 多模态
│   │   └── StructuredOutputService.java       # 结构化输出
│   ├── tool/
│   │   ├── WeatherTool.java                   # 天气查询工具
│   │   ├── CalculatorTool.java                # 计算器工具
│   │   └── TimeTool.java                      # 时间/时区工具
│   └── model/
│       ├── ChatRequest.java                   # 请求模型
│       ├── ChatResponse.java                  # 响应模型
│       └── WeatherInfo.java                   # 天气信息模型
├── src/main/resources/
│   ├── application.yml                        # 核心配置
│   ├── static/index.html                      # 学习指南页面
│   ├── rag-docs/                              # RAG 测试文档
│   └── prompt-templates/                      # Prompt 模板
└── README.md
```

## 🚀 快速开始

### 1. 获取 API Key

访问 [阿里云百炼平台](https://bailian.console.aliyun.com/)，开通模型服务并创建 API Key。

### 2. 配置 API Key

编辑 `src/main/resources/application.yml`：

```yaml
spring:
  ai:
    dashscope:
      api-key: ${DASHSCOPE_API_KEY:your-api-key-here}
```

或设置环境变量：

```bash
export DASHSCOPE_API_KEY=sk-your-api-key-here
```

### 3. 启动项目

```bash
cd ai-learn-spring-ai-alibaba
./mvnw spring-boot:run
```

### 4. 打开学习指南

浏览器访问：`http://localhost:8080`

### 5. 测试接口

```bash
# 基础对话（ChatClient，推荐）
curl -X POST http://localhost:8080/api/chat/client \
  -H "Content-Type: application/json" \
  -d '{"message":"什么是 Spring AI Alibaba？"}'

# 流式输出
curl -N http://localhost:8080/api/stream/chat?message=介绍通义千问

# 多轮对话（带记忆）
curl -X POST http://localhost:8080/api/memory/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"我叫张三"}'

# RAG 知识库问答
curl -X POST http://localhost:8080/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"message":"阿里云百炼平台如何计费？"}'

# Agent 工具调用
curl -X POST http://localhost:8080/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"杭州天气如何？365*24等于多少？"}'

# 结构化输出
curl -X POST http://localhost:8080/api/structured/blog-post \
  -H "Content-Type: application/json" \
  -d '{"topic":"Spring AI Alibaba入门"}'
```

## 💡 学习建议

1. **按顺序学**：第1课 → 第2课 → 第3课 → 第4课 → 第5课 → 第6课 → 第7课
2. **先看注释**：每个类的 Javadoc 都是精心编写的教学文档，比网上任何教程都详细
3. **做对比实验**：对比三种调用方式、有无 RAG、有无记忆的效果差异
4. **动手改参数**：修改 Temperature、Top-P、模型名称，观察输出变化
5. **换模型测试**：将 qwen-plus 换成 qwen-turbo 或 qwen-max 体验差异
6. **对比学习**：与 ai-learn（OpenAI版）对比，理解不同平台的 API 差异

## 🔗 参考资源

- [Spring AI Alibaba 官方文档](https://java2ai.com)
- [阿里云百炼平台](https://bailian.console.aliyun.com/)
- [Spring AI 官方文档](https://docs.spring.io/spring-ai/reference/)
- [通义千问模型介绍](https://help.aliyun.com/zh/model-studio/)
