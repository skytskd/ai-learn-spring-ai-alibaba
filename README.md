# 🧠 ai-learn-spring-ai-alibaba — AI 应用开发 Java 进阶学习项目

基于 **Spring Boot 3.4.5 + Spring AI 1.1.0 + Spring AI Alibaba 1.1.0.0** 的 AI 应用开发项目，使用阿里云通义千问系列模型。

本项目分两个阶段：

- **第 1~7 课：基础能力** —— 打通从对话到 Agent 的完整链路
- **第 8~11 课：深水区** —— RAG 调优、多轮记忆、Agent 编排、评估体系

---

## 🎯 与 ai-learn（OpenAI 版）的区别

| 对比维度 | ai-learn (OpenAI) | ai-learn-spring-ai-alibaba |
|---------|-------------------|---------------------------|
| AI 模型 | OpenAI (GPT-4o 等) | 通义千问 (qwen 系列) |
| API 平台 | OpenAI API | 阿里云百炼 DashScope |
| 国内访问 | 需要代理 | 直接访问，低延迟 |
| 中文能力 | 较好 | 更强（原生中文优化） |
| 多模态 | 单独 API | 通义 VL 原生支持 |

---

## 📚 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Spring Boot | 3.4.5 | 基础框架 |
| Spring AI | 1.1.0 | AI 标准接口层 |
| Spring AI Alibaba | 1.1.0.0 | 阿里云模型集成 |
| Spring WebFlux | 3.4.5 | SSE 流式接口 |
| Lombok | latest | 减少样板代码 |

---

## 🗺️ 完整学习路线

### 第一阶段：基础能力（第 1~7 课）

| 课程 | 模块 | 核心知识点 | 接口 |
|------|------|-----------|------|
| 第 1 课 | BasicChatService | ChatModel、PromptTemplate、ChatClient 三种调用方式 | `POST /api/chat/client` |
| 第 2 课 | StreamChatService | Flux、SSE、响应式流 | `GET /api/stream/chat` |
| 第 3 课 | MemoryChatService | ChatMemory、MessageChatMemoryAdvisor | `POST /api/memory/chat` |
| 第 4 课 | RagService（基础版） | 文档加载→分块→Embedding→向量检索 | `POST /api/rag/ask-baseline` |
| 第 5 课 | AgentService | @Tool、Function Calling、ReAct 模式 | `POST /api/agent/chat` |
| 第 6 课 | MultimodalService | 通义 VL 图片理解 | `POST /api/multimodal/analyze` |
| 第 7 课 | StructuredOutputService | Bean 映射、结构化 JSON 输出 | `POST /api/structured/blog-post` |

### 第二阶段：深水区（第 8~11 课）

| 课程 | 模块 | 核心知识点 | 关键接口 |
|------|------|-----------|---------|
| 第 8 课 | **RAG 调优** | 混合检索、RRF 融合、重排序、HyDE、查询扩展 | `POST /api/rag/ask` `POST /api/rag/explain` |
| 第 9 课 | **多轮记忆** | 滑动窗口、滚动摘要、长期事实、空闲清理 | `POST /api/memory/chat` `GET /api/memory/state` |
| 第 10 课 | **Agent 编排** | 显式 ReAct 循环、迭代上限、死循环检测、轨迹可观测 | `POST /api/agent/chat-trace` |
| 第 11 课 | **评估体系** | RAGAS 四指标、Recall@K/MRR/NDCG、金标准测试集 | `POST /api/eval/run` |

---

## 🔬 第 8 课：RAG 调优

### 为什么基础版 RAG 不够用？

原版只有「向量检索 + 拼接上下文」，在真实场景有三个硬伤：

1. **专有名词检索失效** —— `text-embedding-v2`、`ERR_403` 这类标识符
   被向量模型压成模糊的语义点，无法精确匹配
2. **单查询覆盖不全** —— 一个查询只能覆盖一个语义视角，容易漏召回
3. **无关内容污染上下文** —— 把 Top-K 全塞给 LLM，稀释注意力、诱发幻觉

### 强化版流水线

```
用户问题
   │
   ├─[1] 查询改写（可选）
   │      ├─ 多查询扩展：生成 N 个语义变体，从不同角度召回
   │      └─ HyDE：生成假想答案文档，用其向量检索（问题与答案在向量空间距离远）
   │
   ├─[2] 多路召回
   │      ├─ 稠密检索（向量）→ 语义匹配强
   │      └─ 稀疏检索（BM25）→ 专有名词/编号/代码符号强
   │
   ├─[3] RRF 融合（Reciprocal Rank Fusion）
   │      └─ 只看排名不看分数：免调参、抗异常、奖励双路共识
   │
   └─[4] 重排序（Rerank）
          └─ Cross-Encoder 精排：Top-20 → Top-5
                 │
                 ▼
           拼接上下文 → LLM 生成 → 带引用的回答
```

### 关键设计说明

**RRF 为什么不用分数相加？**

稠密分数（余弦相似度 0.6~0.9）与 BM25 分数（无界）量纲完全不同，
直接加权求和需要人工调参且换数据集就失效。RRF 完全抛弃分数、只用排名：

```
             1
RRF(d) = Σ ──────────
         r  k + rank_r(d)

k = 60（原始论文经验值，控制排名差异的衰减速度）
```

**「召回求全，精排求准」**

这是信息检索的黄金法则。召回阶段宁滥勿缺，精排阶段严格筛选——
因为 LLM 上下文窗口有限，塞入无关内容会直接推高幻觉率。

### 对比实验方法

```bash
# 强化版
curl -X POST http://localhost:8080/api/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"message":"text-embedding-v2 的计费标准如何？"}'

# 基础版（对照）
curl -X POST http://localhost:8080/api/rag/ask-baseline \
  -H "Content-Type: application/json" \
  -d '{"message":"text-embedding-v2 的计费标准如何？"}'

# 检索诊断：看清每一路召回了什么、排名如何变化
curl -X POST http://localhost:8080/api/rag/explain \
  -H "Content-Type: application/json" \
  -d '{"message":"text-embedding-v2 的计费标准如何？"}'
```

---

## 🧠 第 9 课：多轮记忆

### 三层记忆架构

```
┌─────────────────────────────────────────────────────┐
│  Layer 3  长期事实（跨会话稳定）                      │
│  例："用户叫张三"、"用户是 Java 开发工程师"            │
├─────────────────────────────────────────────────────┤
│  Layer 2  会话摘要（压缩控制 token）                  │
│  例："用户询问了 Spring AI 的 RAG 实现方式..."        │
├─────────────────────────────────────────────────────┤
│  Layer 1  近期消息窗口（完整细节）                    │
│  例："用户: 我叫张三" / "助手: 你好张三！"             │
└─────────────────────────────────────────────────────┘
```

### 为什么必须分层？

把全部历史原样塞进上下文是最贵也最差的做法：

- **成本**：每问一次都要重发全部历史，成本随轮次**线性增长**
- **效果**：无关历史稀释注意力，导致「越聊越傻」

分层的本质是**信息分级存储**：近期细节保原文，中期要点压摘要，稳定属性抽事实。

### 滚动摘要的关键

摘要不是「重新总结全部历史」，而是**「旧摘要 + 新溢出内容 → 新摘要」**。
这样每次摘要输入规模恒定，不随对话轮次增长。

配合两重节流（窗口占用触发 + 轮次间隔），避免每轮都调 LLM 做摘要。

### 查看记忆内部状态

```bash
# 第一轮
curl -X POST http://localhost:8080/api/memory/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"我叫张三，是一名Java开发工程师"}'
# → 返回 conversationId，后续请求带上它

# 查看三层记忆状态（调试利器）
curl "http://localhost:8080/api/memory/state?conversationId=<你的ID>"
```

---

## 🤖 第 10 课：Agent 编排

### 核心风险：成本失控

普通 Chat 一次请求 = 一次 LLM 调用，成本可预测。
Agent 一次请求 = N 轮「模型调用 + 工具执行」。

更糟的是：**每轮都要重发全部历史，成本是 O(n²) 而非 O(n)**。
这是 ReAct 成本爆炸的根本原因。

### 三重安全阀

| 防护 | 说明 |
|------|------|
| **硬性迭代上限** | `max-iterations` 轮后强制终止并降级回答 |
| **死循环检测** | 连续 N 次调用相同工具且参数相同 → 判定死循环 |
| **格式纠错重试** | 输出格式不合规时追加提醒，而非直接失败 |

### 为什么要有显式轨迹？

Agent 回答错误时，没有轨迹你只能看到「答案不对」。
有轨迹后能定位到具体环节：

- 没选对工具 → 工具描述有问题
- 工具对了参数错 → 参数 schema 描述不清
- 工具执行失败 → 工具实现有 bug
- 拿到正确结果但推理错 → Prompt 或模型能力问题

这四种情况**外部现象完全一样**，只有轨迹能区分。

### 使用

```bash
# ReAct 模式，返回结构化轨迹
curl -X POST http://localhost:8080/api/agent/chat-trace \
  -H "Content-Type: application/json" \
  -d '{"message":"杭州天气如何？123*456等于多少？"}'

# 查看可用工具
curl http://localhost:8080/api/agent/tools
```

### 工具设计三原则

1. **数量**：控制在 5~8 个。超过 10 个后模型选错率显著上升
2. **描述**：说明「**何时使用**」而非仅「做什么」
   - ❌ "查询用户信息"
   - ✅ "根据用户 ID 查询姓名、邮箱和注册时间。当用户询问自己的账号信息时使用。"
3. **粒度**：一个工具做一件事

---

## 📊 第 11 课：评估体系

### 没有度量就没有优化

RAG 调优最常见的失败模式：改个参数，试两个问题，感觉「好像好了」，
就认定改进有效。这是**过拟合直觉**——无法区分真实提升与随机波动。

### 评估必须分层

如果只评最终答案，效果不好时你无法知道是：

- 检索没找到正确文档（**检索问题**）→ 调切分、召回、重排
- 检索对了但模型没用（**生成问题**）→ 调 Prompt、换模型

这两类问题解法完全不同。混在一起评等于放弃归因能力。

### 指标全景

**检索层指标**

| 指标 | 含义 | 为什么重要 |
|------|------|-----------|
| **Recall@K** | 召回率 | **RAG 中最重要的指标**——检索是生成的输入，丢了无法在下游弥补 |
| Precision@K | 精确率 | 衡量召回结果的纯度，低则稀释注意力 |
| MRR | 平均倒数排名 | 关心第一个正确结果排多前 |
| NDCG@K | 归一化折损累计增益 | 最严谨的排序指标，考虑所有相关结果的位置 |

**生成层指标（RAGAS 四指标）**

| 指标 | 含义 | 诊断方向 |
|------|------|---------|
| **Faithfulness** | 忠实度 | **幻觉的直接度量**。回答中能从上下文得到支撑的论断占比 |
| Answer Relevancy | 答案相关性 | 是否切题（答非所问的反面） |
| Context Precision | 上下文精确率 | 相关分片是否排在前面（排序质量） |
| Context Recall | 上下文召回率 | 标准答案的信息覆盖率（唯一需要标准答案的指标） |

### ⚠️ LLM-as-Judge 的已知偏差

参考 Zheng et al., arXiv 2306.05685：

1. **位置偏差** —— 偏好排在前面的选项
2. **冗长偏差** —— 偏好长回答
3. **自我增强偏差** —— 偏好与自己风格相似的输出
4. **分数聚集** —— 总给 7-8 分，区分度低

因此 LLM 评分应**看趋势而非绝对值**。它适合回答「这次改动变好还是变差」，
不适合回答「当前水平是 0.72」。

### 使用

```bash
# 查看测试集内容（零成本）
curl http://localhost:8080/api/eval/dataset

# 冒烟测试：只跑 3 个样本（成本低，验证链路通畅）
curl -X POST http://localhost:8080/api/eval/smoke

# 完整评估（20 样本，约 120~180 次 LLM 调用）
curl -X POST http://localhost:8080/api/eval/run

# 文本格式报告
curl -X POST http://localhost:8080/api/eval/report
```

**成本提醒**：一次完整评估约需 120~180 次 LLM 调用。
不要放进「每次提交触发」的 CI，应用定时任务或手动触发。

---

## 📁 项目结构

```
ai-learn-spring-ai-alibaba/
├── pom.xml
├── src/main/java/com/ailearn/alibaba/
│   ├── AiLearnAlibabaApplication.java
│   ├── config/
│   │   ├── ChatClientConfig.java        # ChatClient / Advisor 配置
│   │   ├── RagProperties.java           # 【第8课】RAG 调优参数
│   │   ├── MemoryProperties.java        # 【第9课】记忆参数
│   │   └── AgentProperties.java         # 【第10课】Agent 安全阀参数
│   ├── rag/                             # 【第8课】RAG 调优组件
│   │   ├── ScoredDocument.java          # 带分数的检索结果
│   │   ├── Bm25SparseRetriever.java     # BM25 稀疏检索
│   │   ├── RrfFusion.java               # RRF 倒数排名融合
│   │   ├── LlmReranker.java             # LLM 重排序
│   │   └── QueryRewriter.java           # 查询改写 + HyDE
│   ├── memory/                          # 【第9课】分层记忆
│   │   ├── MemoryMessage.java           # 消息载体
│   │   ├── ConversationState.java       # 三层会话状态
│   │   └── LayeredMemoryService.java    # 分层记忆服务
│   ├── agent/                           # 【第10课】Agent 编排
│   │   ├── AgentTrace.java              # 执行轨迹
│   │   └── ReActAgent.java              # 显式 ReAct 循环
│   ├── eval/                            # 【第11课】评估体系
│   │   ├── EvalSample.java              # 评估样本
│   │   ├── GoldenTestSet.java           # 金标准测试集（20 条）
│   │   ├── RetrievalMetrics.java        # Recall@K / MRR / NDCG
│   │   ├── RagasMetrics.java            # RAGAS 四指标
│   │   ├── EvalService.java             # 评估执行器
│   │   └── EvalReport.java              # 评估报告
│   ├── service/
│   │   ├── BasicChatService.java        # 第1课
│   │   ├── StreamChatService.java       # 第2课
│   │   ├── MemoryChatService.java       # 第3课
│   │   ├── RagService.java              # 第4/8课（已强化）
│   │   ├── AgentService.java            # 第5/10课（已强化）
│   │   ├── MultimodalService.java       # 第6课
│   │   └── StructuredOutputService.java # 第7课
│   ├── controller/                      # REST 接口层
│   ├── tool/                            # 工具（WeatherTool / CalculatorTool / TimeTool）
│   │   └── ↑ 需在 config/ToolConfig.java 中注册才生效
│   └── model/
├── src/main/resources/
│   ├── application.yml                  # 核心配置（含 RAG/记忆/Agent 参数）
│   ├── static/index.html
│   ├── rag-docs/                        # RAG 知识库文档
│   └── prompt-templates/
└── README.md
```

---

## 🚀 快速开始

### 1. 配置 API Key

```bash
export AI_API_KEY=sk-your-api-key-here
```

或编辑 `src/main/resources/application.yml`：

```yaml
spring:
  ai:
    dashscope:
      api-key: ${AI_API_KEY:your-api-key-here}
```

### 2. 启动

```bash
cd ai-learn-spring-ai-alibaba
mvn spring-boot:run
```

### 3. 打开学习指南

浏览器访问：`http://localhost:8080`

---

## 💡 调参方法论

**这是本项目最重要的部分**——比任何具体参数值都有价值。

### 铁律：每次只改一个参数

同时改多个参数会导致无法归因。你看到指标提升了，
但不知道是哪个改动带来的，甚至可能一个是正贡献、另一个是负贡献。

### 推荐调参顺序（按投入产出比）

1. **先搭评估体系** —— 没有度量就没有优化，`/api/eval/run` 跑通再说
2. **`chunk-size`** —— 影响最大且不可后期弥补
3. **`enable-rerank`** —— 打开即有提升，收益最直接
4. **`rerank-top-n`** —— 直接影响送入 LLM 的上下文质量
5. **`enable-query-expansion` / `enable-hyde`** —— 提升召回覆盖
6. **`dense-top-k` / `sparse-top-k`** —— 融合前调整各通道候选量

### 标准调优流程

```
1. 跑基线：POST /api/eval/run → 记录总体指标
2. 改一个参数（如 chunk-size: 500 → 300）
3. 重启，再跑评估 → 对比
4. 指标提升 → 保留；下降 → 回滚
5. 进入下一个参数
```

### 常见问题速查

| 现象 | 推断原因 | 调整方向 |
|------|---------|---------|
| Recall 低 | 检索没找到正确内容 | 调小 chunk-size、开混合检索、开查询扩展 |
| Recall 高但 Faithfulness 低 | 检索对了但模型编造 | 加强 Prompt 约束、降低 temperature |
| Faithfulness 高但 Relevancy 低 | 不编造但答非所问 | 检查 Prompt 是否明确要求直接回答 |
| Context Precision 低 | 相关分片排序靠后 | 加强重排序 |
| Context Recall 低 | 上下文覆盖不全 | 增加召回路数、开 HyDE |

---

## ⚠️ 生产环境注意事项

本项目为教学目的做了若干简化，上生产前必须处理：

| 项目 | 当前实现 | 生产要求 |
|------|---------|---------|
| 向量存储 | SimpleVectorStore（内存） | Milvus / Qdrant / pgvector |
| 会话记忆 | ConcurrentHashMap（进程内） | Redis + TTL，或数据库冷归档 |
| 重排序 | LLM 打分（每个候选一次调用） | 专用模型（BGE-Reranker / gte-rerank） |
| 分词 | 字符 bigram | 专业分词器（HanLP / IK / jieba） |
| 工具执行 | 无隔离 | 沙箱 + 超时 + 限流 |
| 评估 | 手动触发 | 定时任务 + 指标趋势看板 |
| 可观测性 | 日志 | 接入 Langfuse / Phoenix，全链路埋点 |

---

## 🔧 实战排障笔记

以下是本项目实际运行中遇到的真实问题与修复方案。
这类「环境 / 框架细节」的坑，教程里通常不会讲，但实际开发中遇到的概率极高。

### 问题 1：启动报 `Port xxxxx was already in use`（端口被环境变量劫持）

**现象**

`application.yml` 里明明写的是 `server.port: 8080`，启动日志却显示：

```
Tomcat initialized with port 62456 (http)
...
Web server failed to start. Port 62456 was already in use.
```

去查 62456 端口，发现占用者是 IDE 或宿主进程本身——改配置文件完全无效。

**根因**

Spring Boot 的**宽松绑定（Relaxed Binding）**规则会把环境变量
`SERVER__PORT`（双下划线）映射为配置项 `server.port`，
且**环境变量优先级高于 `application.yml`**。

部分 IDE、容器运行时、宿主进程会自动注入该变量（用于自身端口管理），
于是应用的端口被悄悄改掉，报错信息还把责任推给「端口被占用」，极具误导性。

**排查方法**

```powershell
# 列出所有含 PORT 的环境变量名
[System.Environment]::GetEnvironmentVariables().Keys | Where-Object { $_ -match 'PORT' }

# 查端口占用者
netstat -ano | Select-String ":62456.*LISTENING"
```

**修复**

**方案 A（推荐，本项目已采用）**：在 `application.yml` 里用占位符绕开它。

```yaml
server:
  port: ${APP_SERVER_PORT:8080}
```

这样只有自定义变量 `APP_SERVER_PORT` 才能覆盖，`SERVER__PORT` 彻底失效。

**方案 B**：启动时用命令行参数覆盖——命令行参数优先级最高，
不受任何环境变量影响。

```bash
mvn spring-boot:run -Dspring-boot.run.arguments=--server.port=8080
```

---

### 问题 2：Agent 一个工具都调不动，`/api/agent/tools` 返回 `count=0`

**现象**

`WeatherTool`、`CalculatorTool`、`TimeTool` 都标了 `@Tool` 和 `@Component`，
`AgentService` 也正常注入了 `List<ToolCallback>`，但工具列表始终为空，
Agent 完全不调用工具。

**根因**

**`@Tool` 注解只是「标记」，不会自动把 Bean 转换成可调用工具。**
真正的转换动作必须显式执行——调用 `ToolCallbacks.from(bean)`。

本项目原先缺少这一步，导致 `List<ToolCallback>` 被解析为一个空集合
（Spring 对「集合类型依赖 + 无匹配 Bean」的处理是注入空集合，而非报错，
所以这个问题不会在启动时暴露，只会在运行时表现为「Agent 变傻」）。

**修复**

新增 `config/ToolConfig.java`，集中注册工具：

```java
@Bean
public List<ToolCallback> toolCallbacks(WeatherTool weatherTool,
                                        CalculatorTool calculatorTool,
                                        TimeTool timeTool) {
    return List.of(
            ToolCallbacks.from(weatherTool),
            ToolCallbacks.from(calculatorTool),
            ToolCallbacks.from(timeTool)
    ).stream().flatMap(Arrays::stream).toList();
}
```

修复后启动日志会打印：

```
[工具注册] 已注册 4 个 Agent 工具：[getWeather, calculate, daysBetween, getCurrentTime]
```

**经验**：凡是「框架里莫名不生效」的问题，先确认
**该框架是否需要显式的注册/转换动作**。注解通常只负责声明，
不负责注册——Spring 生态里类似的设计还有 `@ConfigurationProperties`
（需要 `@EnableConfigurationProperties`）、`@Aspect`（需要 `@EnableAspectJAutoProxy`）等。

---

### 问题 3：DashScope 配额耗尽导致应用完全起不来

**现象**

应用启动直接失败，异常栈指向 `RagService.init`：

```
HTTP 403 - {"code":"AllocationQuota.FreeTierOnly",
            "message":"Free quota exhausted. ..."}
```

根因是 `RagService` 在 `@PostConstruct` 里同步调用了 Embedding 接口
去构建向量索引。**把「外部服务可用性」变成了「应用能否启动」的前置条件**，
这是一个典型的设计缺陷——外部依赖抖动时，应用会连带挂掉。

**修复**

改为**惰性构建 + 优雅降级**：

1. `@PostConstruct` 只做**纯本地**工作：文档加载、切分、BM25 倒排索引
   （零网络调用，必然成功）
2. 向量索引改为**首次检索时构建**，用双重检查锁保证线程安全与幂等
3. 向量索引构建失败**不抛异常**，只记录警告，自动降级为**纯 BM25 检索**
4. 提供 `POST /api/rag/rebuild`，补充额度后**无需重启**即可恢复混合检索

修复后启动日志：

```
[RAG] BM25 倒排索引构建完成
[RAG] 向量索引构建失败，降级为纯 BM25 检索：HTTP 403 ...
[RAG] 知识库就绪（BM25 已启用，向量索引=未就绪），分片数 2
Tomcat started on port 8080
Started AiLearnAlibabaApplication in 5.111 seconds
```

**设计原则**：**核心链路不依赖外部服务可用性。**
任何在启动阶段调用远程接口的代码，都应改为惰性初始化 + 降级兜底。
AI 应用尤其要注意这点——模型服务限流、欠费、网络抖动都是常态。

---

## 🔗 参考资源

### 官方文档
- [Spring AI Alibaba 官方文档](https://java2ai.com)
- [Spring AI 官方文档](https://docs.spring.io/spring-ai/reference/)
- [阿里云百炼平台](https://bailian.console.aliyun.com/)

### 核心论文（均为 arXiv 免费全文）

**RAG 方向**
- *Retrieval-Augmented Generation for Large Language Models: A Survey* — arXiv 2312.10997（RAG 全景综述）
- *Precise Zero-Shot Dense Retrieval without Relevance Labels*（HyDE）— arXiv 2212.10496
- *Self-RAG: Learning to Retrieve, Generate, and Critique* — arXiv 2310.11511

**记忆方向**
- *MemGPT: Towards LLMs as Operating Systems* — arXiv 2310.08560（Agent Memory 思想源头）
- *Zep: A Temporal Knowledge Graph Architecture for Agent Memory* — arXiv 2501.13956

**Agent 方向**
- *ReAct: Synergizing Reasoning and Acting in Language Models* — arXiv 2210.03629
- *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation* — arXiv 2308.08155

**评估方向**
- *RAGAS: Automated Evaluation of Retrieval Augmented Generation* — arXiv 2309.15217
- *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* — arXiv 2306.05685（**必读**，讲清 judge 偏差）
- *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment* — arXiv 2303.16634

### 建议的学习方式

**以官方文档和论文为准，视频教程只做入门引子。**
技术迭代快，视频往往滞后半年以上，而论文和官方仓库是同步更新的。
