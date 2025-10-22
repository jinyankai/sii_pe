import json
import re


def construct_prompt(d):
    """
    构造用于大语言模型的提示词 (结构化思维链 Structured CoT - 中文)
    引导模型输出一个包含分析过程的 JSON 对象。

    参数:
    d (dict): jsonl数据文件的一行，解析成字典后的变量。
              注意：传入的 'd' 已经过处理，其 'test' 字段列表
              只包含 'input'，不包含 'output' 答案。

    返回:
    list: OpenAI API的message格式列表
    """

    def format_grid(grid):
        """辅助函数：将网格格式化为字符串，每行一个列表。"""
        return "\n".join([str(row) for row in grid])

    # 1. 系统提示词：设定模型的角色和任务
    system_prompt = """你是一个解决抽象视觉推理谜题的专家，特别是来自 ARC (Abstraction and Reasoning Corpus) 的谜题。
你的目标是分析几个网格变换的训练样本，推导出隐藏的逻辑规则，然后将这个规则精确地应用到一个新的测试输入网格上。
你必须严格按照用户要求的 JSON 结构输出你的完整思考过程和最终答案。
"""

    # 2. 用户提示词：构建 CoT 指导
    user_prompt_lines = []
    user_prompt_lines.append("这里是几个训练样本，请你找出变换规则。")
    user_prompt_lines.append("\n--- 训练样本开始 ---")

    # 2.1. 格式化所有训练样本
    for i, example in enumerate(d['train']):
        input_grid = format_grid(example['input'])
        output_grid = format_grid(example['output'])
        user_prompt_lines.append(f"\n样本 {i + 1} 输入:")
        user_prompt_lines.append(input_grid)
        user_prompt_lines.append(f"样本 {i + 1} 输出:")
        user_prompt_lines.append(output_grid)

    user_prompt_lines.append("\n--- 训练样本结束 ---")

    # 2.2. 提供测试输入
    user_prompt_lines.append("\n现在，这是需要你预测的测试输入网格：")
    user_prompt_lines.append(f"\n测试输入:")
    test_input_grid = format_grid(d['test'][0]['input'])
    user_prompt_lines.append(test_input_grid)

    # 2.3. 提供结构化思维链 (Structured CoT) 指导
    user_prompt_lines.append("\n--- 指导说明 ---")
    user_prompt_lines.append("你的任务是分析训练样本，推导出变换规则，并将该规则应用于“测试输入”。")
    user_prompt_lines.append(
        "请严格按照以下 JSON 结构输出你的完整思考过程和最终答案。不要在 JSON 对象之外添加任何其他文本。")

    # 定义模型必须遵循的 JSON 结构
    json_structure_example = """
**输出 JSON 结构：**
{
  "analysis": {
    "observation_summary": "对所有训练样本的总体观察摘要。",
    "shape_transformation": "分析输入和输出网格之间的形状（尺寸）变化。例如：'保持不变', '从 NxM 变为 PxQ', '裁剪', '扩展'。",
    "element_transformation": "分析元素（颜色/数字）的变化。例如：'颜色 2 变为 4', '所有非零元素变为 1', '背景色 0 保持不变'。",
    "identified_operations": [
      "基于对形状和元素变化的分析，推断出导致变换的具体操作列表。",
      "请重点考虑以下空间操作（但不限于）：",
      "'shift_up' (上移), 'shift_down' (下移), 'shift_left' (左移), 'shift_right' (右移)",
      "'flip_horizontal' (水平反转), 'flip_vertical' (垂直反转)",
      "'transpose' (转置)",
      "'rotate_90_clockwise' (顺时针旋转90度), 'rotate_180' (旋转180度), 'rotate_270_clockwise' (顺时针旋转270度)",
      "'copy_pattern_X' (复制模式X), 'fill_color_Y' (填充颜色Y), 'remove_color_Z' (移除颜色Z)"
    ]
  },
  "rule_derivation": "基于以上分析，用简洁的语言总结出能解释 *所有* 训练样本的变换规则。",
  "rule_application_to_test": "描述如何将这条规则逐步应用于“测试输入”网格，以得到最终的“测试输出”。",
  "predicted_grid": [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
  ]
}
"""
    # 使用 .strip() 来移除开头和结尾的额外空白
    user_prompt_lines.append(json_structure_example.strip())

    user_content = "\n".join(user_prompt_lines)

    # 3. 组装成 OpenAI API 格式
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def parse_output(text):
    """
    解析大语言模型的输出文本，提取预测的网格。
    该版本假定模型输出一个包含 "predicted_grid" 键的 JSON 对象。

    参数:
    text (str): 大语言模型在设计prompt下的输出文本

    返回:
    list: 从输出 JSON 对象中提取的 "predicted_grid" (二维数组)
          如果解析失败，返回空列表 []
    """
    try:
        # 查找从 '{' 开始到 '}' 结束的第一个（也希望是唯一一个）JSON 对象
        # re.DOTALL (re.S) 允许 '.' 匹配换行符
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            json_string = match.group(0)

            # 尝试解析这个字符串为Python字典
            parsed_json = json.loads(json_string)

            # 检查 "predicted_grid" 键是否存在
            if isinstance(parsed_json, dict) and 'predicted_grid' in parsed_json:
                predicted_grid = parsed_json['predicted_grid']

                # 基本的类型验证
                if isinstance(predicted_grid, list):
                    # 确保它是一个列表的列表（或者是空列表）
                    if not predicted_grid or all(isinstance(row, list) for row in predicted_grid):
                        return predicted_grid

    except (json.JSONDecodeError, re.error, Exception) as e:
        # 捕获JSON解析错误、正则错误或任何其他异常
        # print(f"Error parsing output: {e}") # 可以在调试时取消注释
        pass

    # 如果没有找到匹配项，或者解析失败，或者 "predicted_grid" 格式不正确
    return []
