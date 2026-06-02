package com.ailearn.alibaba.tool;

import com.ailearn.alibaba.model.WeatherInfo;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

/**
 * <h1>WeatherTool — 天气查询工具</h1>
 *
 * <p>演示 {@code @Tool} 注解的使用方式。
 * Agent 会自动识别此方法为可调用工具，并传入参数。</p>
 *
 * <h2>@Tool 注解说明</h2>
 * <pre>
 * @Tool(name = "方法名", description = "工具的功能描述（给模型看的）")
 * public ReturnType methodName(
 *     @ToolParam(description = "参数说明（给模型看的）") Type param
 * )
 * </pre>
 *
 * <h2>关键要点</h2>
 * <ul>
 *   <li>{@code description} 要清晰准确——模型根据描述决定是否调用</li>
 *   <li>{@code @ToolParam(description = ...)} 帮助模型理解参数含义</li>
 *   <li>返回类型可以是任意 Java 对象，模型能理解其结构</li>
 * </ul>
 *
 * <h2>模拟数据说明</h2>
 * <p>实际项目中应调用第三方天气 API（如高德、和风天气）。
 * 这里使用模拟数据演示 Function Calling 流程。</p>
 *
 * @author ai-learn
 */
@Slf4j
@Component
public class WeatherTool {

    /**
     * 查询指定城市的天气信息
     *
     * <p>当用户询问天气相关问题时，Agent 会自动调用此方法。
     * 例如："今天杭州天气怎么样？" → Agent 调用 getWeather("杭州")</p>
     *
     * @param city 城市名称，如"杭州"、"北京"、"上海"
     * @return 天气信息对象（城市、天气状况、温度、湿度）
     */
    @Tool(description = "查询指定城市的实时天气信息，返回天气状况、温度和湿度")
    public WeatherInfo getWeather(
            @ToolParam(description = "城市名称，例如：杭州、北京、上海") String city
    ) {
        log.info("[天气工具] 查询城市: {}", city);

        // 模拟天气数据（实际项目调用第三方 API）
        return switch (city) {
            case "杭州" -> new WeatherInfo("杭州", "晴转多云", 25.0, 65);
            case "北京" -> new WeatherInfo("北京", "多云", 22.0, 40);
            case "上海" -> new WeatherInfo("上海", "小雨", 20.0, 80);
            case "深圳" -> new WeatherInfo("深圳", "晴", 30.0, 70);
            case "成都" -> new WeatherInfo("成都", "阴", 18.0, 75);
            default    -> new WeatherInfo(city, "晴", 23.0, 55);
        };
    }
}
