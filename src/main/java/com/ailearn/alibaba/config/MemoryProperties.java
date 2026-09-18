package com.ailearn.alibaba.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * <h1>MemoryProperties — 多轮记忆可调参数</h1>
 *
 * <p>分层记忆的核心参数集中在这里。每个参数都直接影响
 * 「上下文 token 成本」与「记忆保真度」之间的权衡。</p>
 *
 * <h2>核心权衡</h2>
 * <pre>
 *   记忆越完整 ──────────────────────→ 记忆越精简
 *   （保真度高，token 成本高）        （省 token，但可能遗忘）
 *
 *        maxRecentMessages  ← 调大往左，调小往右
 *        summarizeThreshold ← 调大往左（更晚压缩），调小往右
 * </pre>
 *
 * @author ai-learn
 */
@Data
@Component
@ConfigurationProperties(prefix = "ai-learn.memory")
public class MemoryProperties {

    /**
     * Layer 1 滑动窗口保留的消息条数。
     *
     * <p>注意这里统计的是「消息」而非「轮次」，且用户消息与助手回复
     * 各算一条。因此 {@code maxRecentMessages = 10} 约等于 5 轮完整对话。</p>
     *
     * <p>经验区间 8 ~ 20：</p>
     * <ul>
     *   <li>太小（&lt; 6）：刚刚说过的事就忘了，多轮体验差</li>
     *   <li>太大（&gt; 30）：每轮重发的 token 量大，成本与延迟显著上升，
     *       且长上下文容易让模型注意力涣散</li>
     * </ul>
     */
    private int maxRecentMessages = 12;

    /**
     * 触发摘要压缩的消息数阈值。
     *
     * <p>当窗口内消息数超过此值时，才可能触发摘要。
     * 必须 ≥ {@code maxRecentMessages}，否则会频繁触发摘要。</p>
     */
    private int summarizeThreshold = 20;

    /**
     * 两次摘要之间的最小轮次间隔。
     *
     * <p>摘要本身要调用一次 LLM，成本不低。若每轮都触发，
     * 对话延迟会翻倍。这个参数是「记忆新鲜度」与「性能」的折中。</p>
     *
     * <p>经验值 3 ~ 5 轮。</p>
     */
    private int summaryInterval = 3;

    /**
     * 空闲会话的存活时长（分钟）。
     *
     * <p>超过此时长未访问的会话将被清理，防止进程内存储无限增长。
     * 默认 120 分钟（2 小时）。</p>
     *
     * <p>⚠️ 这个值设得过大是内存泄漏的常见成因；
     * 设得过小则用户离开一会儿回来发现记忆丢失。</p>
     */
    private long idleTtlMinutes = 120;

    /**
     * 是否启用长期事实抽取（Layer 3）。
     *
     * <p>抽取需要额外一次 LLM 调用，但收益明显——
     * 用户偏好和身份信息可以跨会话稳定生效。</p>
     */
    private boolean enableFactExtraction = true;
}
