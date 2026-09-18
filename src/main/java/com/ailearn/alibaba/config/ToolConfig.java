package com.ailearn.alibaba.config;

import com.ailearn.alibaba.tool.CalculatorTool;
import com.ailearn.alibaba.tool.TimeTool;
import com.ailearn.alibaba.tool.WeatherTool;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.support.ToolCallbacks;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

/**
 * <h1>ToolConfig — Agent 工具注册中心</h1>
 *
 * <p>把标注了 {@code @Tool} 的普通 Spring Bean 转换成 Spring AI 能够识别的
 * {@link ToolCallback}，并集中注册为 Bean，供 {@code AgentService} 注入使用。</p>
 *
 * <h2>为什么不能自动生效？</h2>
 * <p>这是一个非常容易踩的坑：给方法加 {@code @Tool} 注解、给类加 {@code @Component}
 * 之后，<b>并不会自动变成可调用工具</b>。{@code @Tool} 只是一个「标记」，
 * 真正的转换动作需要显式执行——即调用 {@link ToolCallbacks#from(Object...)}。</p>
 *
 * <p>本项目最初就是因为缺少这一步，导致 {@code AgentService} 注入
 * {@code List<ToolCallback>} 时拿到的是空列表，Agent 一个工具都调不动，
 * 而 {@code /api/agent/tools} 返回 {@code count=0}。</p>
 *
 * <h2>两种注册方式的区别</h2>
 * <table border="1">
 *   <tr><th>方式</th><th>写法</th><th>适用场景</th></tr>
 *   <tr><td>静态注册（本类）</td>
 *       <td>{@code ToolCallbacks.from(bean)} 转成 Bean</td>
 *       <td>工具集合固定、需要统一注入与集中管理</td></tr>
 *   <tr><td>动态注册</td>
 *       <td>{@code chatClient.prompt().tools(bean)} 直接传对象</td>
 *       <td>不同请求使用不同工具子集</td></tr>
 * </table>
 *
 * <h2>工具设计的取舍</h2>
 * <p>工具并非越多越好。工具描述全部挤在同一个 prompt 中，数量超过 10 个后
 * 模型选择错误的概率会显著上升。业界经验是控制在 <b>5 ~ 8 个</b>，
 * 且每个工具只做一件事。若确实需要大量能力，应改用「工具分组 +
 * 按意图路由」的策略，而非一次性全量暴露。</p>
 *
 * @author ai-learn
 */
@Slf4j
@Configuration
public class ToolConfig {

    /**
     * 注册全部 {@code @Tool} 工具为 {@link ToolCallback}。
     *
     * <p>用 {@link ToolCallbacks#from(Object...)} 一次性转换多个工具对象。
     * 返回 {@code List<ToolCallback>} 后，Spring 会把列表中的每个元素
     * 都作为独立 Bean 注册，因此 {@code AgentService} 里注入
     * {@code List<ToolCallback>} 能拿到全部工具。</p>
     *
     * @param weatherTool    天气查询工具
     * @param calculatorTool 计算器工具
     * @param timeTool       时间查询工具
     * @return 工具回调列表
     */
    @Bean
    public List<ToolCallback> toolCallbacks(WeatherTool weatherTool,
                                            CalculatorTool calculatorTool,
                                            TimeTool timeTool) {
        List<ToolCallback> callbacks = List.of(
                ToolCallbacks.from(weatherTool),
                ToolCallbacks.from(calculatorTool),
                ToolCallbacks.from(timeTool)
        ).stream()
                .flatMap(java.util.Arrays::stream)
                .toList();

        log.info("[工具注册] 已注册 {} 个 Agent 工具：{}",
                callbacks.size(),
                callbacks.stream()
                        .map(cb -> cb.getToolDefinition().name())
                        .toList());

        if (callbacks.size() > 8) {
            log.warn("[工具注册] 工具数量 {} 已超过经验建议上限 8 个，"
                    + "模型选错工具的概率会上升，建议按意图分组路由", callbacks.size());
        }

        return callbacks;
    }
}
