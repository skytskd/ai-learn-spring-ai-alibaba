package com.ailearn.alibaba.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * <h1>WeatherInfo — 天气信息模型</h1>
 *
 * <p>Agent 工具调用时返回的天气数据结构。
 * {@code @Tool} 方法通过此对象向大模型描述天气查询结果。</p>
 *
 * @author ai-learn
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class WeatherInfo {

    /** 城市名称 */
    private String city;

    /** 天气状况（晴、多云、雨等） */
    private String condition;

    /** 温度（摄氏度） */
    private double temperature;

    /** 湿度（百分比） */
    private int humidity;

    public String toMarkdown() {
        return String.format("""
                📍 **%s** 天气报告
                - 天气：%s
                - 温度：%.1f°C
                - 湿度：%d%%
                """, city, condition, temperature, humidity);
    }
}
