package com.ailearn.alibaba.eval;

import org.springframework.stereotype.Component;

import java.util.List;

/**
 * <h1>GoldenTestSet — 金标准测试集</h1>
 *
 * <p>评估体系的数据基础。所有问题都针对本项目
 * {@code resources/rag-docs/} 下的知识库文档设计。</p>
 *
 * <h2>测试集设计原则</h2>
 * <ol>
 *   <li><b>覆盖不同难度</b>：事实提取（简单）→ 跨文档对比（中等）→
 *       归纳推理（困难）</li>
 *   <li><b>包含"知识库外"样本</b>：用于验证系统是否会诚实地回答
 *       "资料中未涉及"，而不是胡编。这是幻觉防护的关键测试。</li>
 *   <li><b>包含专有名词查询</b>：如 {@code text-embedding-v2}、
 *       {@code qwen-vl-plus}，用于验证混合检索相比纯向量检索的优势。</li>
 *   <li><b>分类标注</b>：便于分析"哪一类问题表现最差"，
 *       从而定向优化。</li>
 * </ol>
 *
 * <h2>关于测试集规模</h2>
 * <p>本类内置 20 条样本，用于教学演示与快速回归。真实项目中
 * 建议扩充到 50 ~ 100 条，且<b>必须来自真实用户提问</b>——
 * 自己编的问题往往比真实问题更"规整"，会高估系统表现。</p>
 *
 * <h2>维护约定</h2>
 * <p>知识库文档变更后，必须同步检查测试集的
 * {@code keyPhrases} 是否仍然成立。这是测试集最容易腐化的地方。</p>
 *
 * @author ai-learn
 */
@Component
public class GoldenTestSet {

    /**
     * 返回全部金标准样本。
     */
    public List<EvalSample> all() {
        return List.of(
                // ========== 分类 1：计费相关（事实提取）==========
                EvalSample.of("q01",
                        "阿里云百炼平台的计费方式是什么？",
                        "百炼平台采用按量付费（Pay-as-you-go）模式，按实际使用的 Token 数量计费。",
                        "计费",
                        "按量付费", "Pay-as-you-go"),

                EvalSample.of("q02",
                        "qwen-turbo 模型的价格是多少？",
                        "qwen-turbo 约 ¥0.3/千Token（输入），¥0.6/千Token（输出）。",
                        "计费",
                        "qwen-turbo", "0.3", "0.6"),

                EvalSample.of("q03",
                        "qwen-max 的定价比 qwen-turbo 贵多少？",
                        "qwen-max 输入约 ¥20/千Token、输出约 ¥60/千Token；"
                                + "qwen-turbo 输入约 ¥0.3/千Token、输出约 ¥0.6/千Token，"
                                + "输入价格相差约 66 倍，输出价格相差约 100 倍。",
                        "计费",
                        "qwen-max", "qwen-turbo"),

                EvalSample.of("q04",
                        "text-embedding-v2 这个模型的计费标准如何？",
                        "text-embedding-v2 约 ¥0.5/千Token。",
                        "计费",
                        "text-embedding-v2", "0.5"),

                // ========== 分类 2：模型能力（专有名词查询）==========
                EvalSample.of("q05",
                        "百炼平台有哪些可用的对话模型？",
                        "包括 qwen-turbo（速度最快）、qwen-plus（性价比最高）、"
                                + "qwen-max（能力最强）、qwen-long（支持超长上下文）。",
                        "模型",
                        "qwen-long", "qwen-plus", "qwen-turbo"),

                EvalSample.of("q06",
                        "qwen-long 有什么特殊能力？",
                        "qwen-long 支持超长上下文，达到千万 Token 级别。",
                        "模型",
                        "qwen-long", "超长上下文"),

                EvalSample.of("q07",
                        "qwen-vl-max 和 qwen-vl-plus 有什么区别？",
                        "两者都是视觉理解模型，qwen-vl-plus 性价比更高，"
                                + "qwen-vl-max 提供最强的视觉理解能力。",
                        "模型",
                        "qwen-vl-max", "qwen-vl-plus"),

                EvalSample.of("q08",
                        "哪个视觉模型适合日常使用？",
                        "qwen-vl-plus 是性价比推荐的视觉模型。",
                        "模型",
                        "qwen-vl-plus"),

                // ========== 分类 3：平台使用（流程类）==========
                EvalSample.of("q09",
                        "如何获取百炼平台的 API Key？",
                        "访问 https://bailian.console.aliyun.com/ ，"
                                + "在「模型服务 → API Key 管理」中创建。",
                        "平台使用",
                        "API Key", "bailian.console.aliyun.com"),

                EvalSample.of("q10",
                        "新用户使用百炼平台有免费额度吗？",
                        "新用户通常有免费试用额度，大约 100 万 Token。",
                        "平台使用",
                        "免费", "100 万"),

                EvalSample.of("q11",
                        "百炼平台需要先注册什么账号？",
                        "需要先注册阿里云账号，然后开通百炼服务。",
                        "平台使用",
                        "阿里云", "注册"),

                // ========== 分类 4：Spring AI Alibaba 架构（理解类）==========
                EvalSample.of("q12",
                        "Spring AI Alibaba 的架构分为哪几层？",
                        "分为三层：接口层（Spring AI 标准化 API，如 ChatClient、ChatModel、"
                                + "EmbeddingModel）、实现层（DashScope 适配器，把标准 API 请求"
                                + "转换为 DashScope 调用）、平台层（阿里云百炼平台）。",
                        "架构",
                        "接口层", "实现层", "平台层"),

                EvalSample.of("q13",
                        "Spring AI Alibaba 的核心特性有哪些？",
                        "包括：多模型支持（通义千问/万相/VL）、Spring 生态集成、"
                                + "标准化接口、企业级能力（RAG/Agent/Function Calling）、中文优化。",
                        "架构",
                        "多模型", "标准化接口"),

                EvalSample.of("q14",
                        "相比直接使用 OpenAI，Spring AI Alibaba 有什么优势？",
                        "优势包括：国内访问速度快延迟低、中文理解和生成能力更强、"
                                + "符合国内合规要求、成本更低、与阿里云生态集成。",
                        "架构",
                        "国内访问", "合规"),

                EvalSample.of("q15",
                        "Spring AI Alibaba 是怎么让 Java 开发者使用 AI 能力的？",
                        "通过注解、自动配置和依赖注入的方式，让开发者可以像使用 Spring MVC 一样使用 AI 能力。",
                        "架构",
                        "注解", "自动配置", "依赖注入"),

                // ========== 分类 5：跨文档归纳（困难）==========
                EvalSample.of("q16",
                        "如果我要做一个既便宜、中文效果又好的问答系统，应该选哪个对话模型？为什么？",
                        "推荐 qwen-plus，它是性价比最高的选择（约 ¥2/千Token 输入、"
                                + "¥6/千Token 输出），且通义系列对中文有原生优化。"
                                + "若预算极度紧张可用 qwen-turbo。",
                        "综合",
                        "qwen-plus", "中文"),

                EvalSample.of("q17",
                        "搭建 RAG 系统需要用到哪些模型？",
                        "需要两类模型：对话模型（如 qwen-plus）负责生成回答，"
                                + "Embedding 模型（text-embedding-v2）负责把文本向量化以支持检索。",
                        "综合",
                        "text-embedding-v2", "Embedding"),

                // ========== 分类 6：知识库外（幻觉防护测试）==========
                // 这类样本最重要：系统应当诚实回答"资料中未涉及"，
                // 而不是编造答案。若这些样本的 Faithfulness 很低，
                // 说明系统在编造内容。
                EvalSample.of("q18",
                        "百炼平台的服务器部署在哪个城市？",
                        "资料中未涉及此信息。",
                        "知识库外",
                        "未提及"),

                EvalSample.of("q19",
                        "通义千问模型有多少个参数？",
                        "资料中未涉及具体的参数规模信息。",
                        "知识库外",
                        "未提及"),

                EvalSample.of("q20",
                        "百炼平台支持支付宝付款吗？",
                        "资料中未涉及支付方式的说明。",
                        "知识库外",
                        "未提及")
        );
    }

    /**
     * 按分类筛选样本，用于分组分析。
     *
     * <p>整体指标往往掩盖问题：例如总分 0.8 但「专有名词」类
     * 只有 0.3。分组看才能发现这类结构性短板。</p>
     */
    public List<EvalSample> byCategory(String category) {
        return all().stream()
                .filter(s -> category.equals(s.getCategory()))
                .toList();
    }

    /** 全部分类标签 */
    public List<String> categories() {
        return all().stream()
                .map(EvalSample::getCategory)
                .distinct()
                .toList();
    }

    /** 测试集规模 */
    public int size() {
        return all().size();
    }
}
