package com.ailearn.alibaba.tool;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;

/**
 * <h1>TimeTool — 时间查询工具</h1>
 *
 * <p>给 Agent 提供时间和时区查询能力。
 * 演示不同时区的时间转换。</p>
 *
 * @author ai-learn
 */
@Slf4j
@Component
public class TimeTool {

    private static final DateTimeFormatter FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss EEEE");

    /**
     * 获取指定时区的当前时间
     *
     * @param timezone 时区（如"Asia/Shanghai"、"America/New_York"）
     * @return 格式化的时间字符串
     */
    @Tool(description = "获取指定时区的当前日期和时间")
    public String getCurrentTime(
            @ToolParam(description = "时区，如：Asia/Shanghai（中国）、"
                    + "America/New_York（纽约）、Europe/London（伦敦）")
            String timezone
    ) {
        log.info("[时间工具] 查询时区: {}", timezone);

        try {
            ZoneId zone = ZoneId.of(timezone);
            ZonedDateTime now = ZonedDateTime.now(zone);
            return now.format(FORMATTER) + " (" + timezone + ")";
        } catch (Exception e) {
            // 如果时区无效，返回默认的中国时区
            ZonedDateTime chinaTime = ZonedDateTime.now(ZoneId.of("Asia/Shanghai"));
            return chinaTime.format(FORMATTER) + " (Asia/Shanghai)";
        }
    }

    /**
     * 计算两个日期之间的天数差
     *
     * @param date1 日期1（格式：yyyy-MM-dd）
     * @param date2 日期2（格式：yyyy-MM-dd）
     * @return 天数差
     */
    @Tool(description = "计算两个日期之间相隔多少天")
    public String daysBetween(
            @ToolParam(description = "第一个日期，格式：yyyy-MM-dd") String date1,
            @ToolParam(description = "第二个日期，格式：yyyy-MM-dd") String date2
    ) {
        log.info("[时间工具] 日期差: {} ~ {}", date1, date2);

        try {
            LocalDateTime d1 = LocalDateTime.parse(date1 + "T00:00:00");
            LocalDateTime d2 = LocalDateTime.parse(date2 + "T00:00:00");
            long days = Math.abs(java.time.Duration.between(d1, d2).toDays());
            return String.format("%s 到 %s 相差 %d 天", date1, date2, days);
        } catch (Exception e) {
            return "日期格式错误，请使用 yyyy-MM-dd 格式";
        }
    }
}
