package com.ailearn.alibaba.tool;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

/**
 * <h1>CalculatorTool — 计算器工具</h1>
 *
 * <p>给 Agent 提供数学计算能力。演示如何处理简单的工具调用。</p>
 *
 * <h2>设计要点</h2>
 * <ul>
 *   <li><b>单一职责</b>：每个工具只做一件事</li>
 *   <li><b>输入校验</b>：检查参数有效性，避免除零等错误</li>
 *   <li><b>错误提示</b>：清晰的错误信息帮助模型调整策略</li>
 * </ul>
 *
 * @author ai-learn
 */
@Slf4j
@Component
public class CalculatorTool {

    /**
     * 执行基本数学运算
     *
     * <p>支持加减乘除四则运算。</p>
     *
     * @param expression 数学表达式，如 "123*456"、"(100+200)/3"
     * @return 计算结果字符串
     */
    @Tool(description = "执行基本的数学运算，支持加减乘除。"
            + "表达式示例：123*456、100+200、100/3")
    public String calculate(
            @ToolParam(description = "数学表达式，如：123*456、100/3") String expression
    ) {
        log.info("[计算器工具] 表达式: {}", expression);

        try {
            // 注意：生产环境应使用专业的表达式引擎（如 exp4j、EvalEx）
            // 这里用简单的字符串解析演示 Agent 工具调用流程

            expression = expression.replaceAll("\\s+", "");

            // 乘法
            if (expression.contains("*")) {
                String[] parts = expression.split("\\*");
                double result = Double.parseDouble(parts[0]) * Double.parseDouble(parts[1]);
                return String.format("%s = %.2f", expression, result);
            }

            // 除法
            if (expression.contains("/")) {
                String[] parts = expression.split("/");
                double denominator = Double.parseDouble(parts[1]);
                if (denominator == 0) {
                    return "错误：除数不能为零";
                }
                double result = Double.parseDouble(parts[0]) / denominator;
                return String.format("%s = %.2f", expression, result);
            }

            // 加法
            if (expression.contains("+")) {
                String[] parts = expression.split("\\+");
                double result = Double.parseDouble(parts[0]) + Double.parseDouble(parts[1]);
                return String.format("%s = %.2f", expression, result);
            }

            // 减法
            if (expression.contains("-")) {
                // 注意：减号也可能是负号，简单处理
                String[] parts = expression.split("-");
                if (parts.length == 2) {
                    double result = Double.parseDouble(parts[0]) - Double.parseDouble(parts[1]);
                    return String.format("%s = %.2f", expression, result);
                }
            }

            return "不支持的表达式格式：" + expression;

        } catch (NumberFormatException e) {
            return "表达式格式错误，请使用数字和 + - * / 运算符";
        }
    }
}
