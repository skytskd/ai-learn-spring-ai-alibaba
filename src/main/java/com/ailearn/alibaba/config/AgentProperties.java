package com.ailearn.alibaba.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * <h1>AgentProperties — Agent 编排可调参数</h1>
 *
 * <p>Agent 的最大风险是<b>成本失控</b>。普通 Chat 一次请求就是一次
 * LLM 调用，成本可预测；Agent 一次请求可能触发 N 轮
 * 「模型调用 + 工具执行」，成本随迭代次数放大。</p>
 *
 * <p>因此 Agent 的所有安全阀参数都必须是<b>硬约束</b>，
 * 绝不能依赖模型的自觉。</p>
 *
 * <h2>成本估算</h2>
 * <pre>
 *   单次请求成本 ≈ maxIterations × (prompt_tokens + completion_tokens) × 单价
 *
 *   注意 prompt 会随迭代轮次<b>累积增长</b>——每轮都要重发全部历史，
 *   因此实际成本是 O(n²) 而非 O(n)。这是 ReAct 成本爆炸的根本原因。
 * </pre>
 *
 * @author ai-learn
 */
@Data
@Component
@ConfigurationProperties(prefix = "ai-learn.agent")
public class AgentProperties {

    /**
     * 最大迭代轮数（硬性安全阀）。
     *
     * <p>超过此轮数会强制终止并降级回答。<b>这个值必须设置</b>——
     * 没有上限的 Agent 在生产环境是重大事故隐患。</p>
     *
     * <p>经验区间 5 ~ 10：</p>
     * <ul>
     *   <li>太小（&lt; 3）：复杂任务无法完成，工具还没调用完就被中断</li>
     *   <li>太大（&gt; 15）：成本失控风险高，且长上下文会降低模型表现</li>
     * </ul>
     *
     * <p>注意：一次「工具调用」消耗一轮迭代。因此 maxIterations=8
     * 大致支持「调用 6 个工具 + 1 轮总结」。</p>
     */
    private int maxIterations = 8;

    /**
     * 是否启用显式 ReAct 循环。
     *
     * <ul>
     *   <li>true：使用本项目的 {@code ReActAgent}，中间步骤可见、可控</li>
     *   <li>false：使用框架原生 Function Calling，一行代码但不可观测</li>
     * </ul>
     *
     * <p>生产环境推荐根据需求选择：需要严格成本控制与可观测性用 true；
     * 追求开发效率且任务简单用 false。</p>
     */
    private boolean enableReAct = true;

    /**
     * 是否在响应中返回完整执行轨迹。
     *
     * <p>调试时开启，生产环境建议关闭以减少响应体积
     * （轨迹可能包含大量中间文本）。</p>
     */
    private boolean returnTrace = true;

    /**
     * 工具调用超时（秒）。
     *
     * <p>防止某个工具（如外部 HTTP 调用）挂死导致整个请求阻塞。</p>
     */
    private int toolTimeoutSeconds = 30;
}
