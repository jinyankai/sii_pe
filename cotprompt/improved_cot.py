import json
import re


def construct_prompt(d):
    """
    构造用于大语言模型的提示词 (结构化CoT + 抽象视图)

    参数:
    d (dict): jsonl数据文件的一行，解析成字典后的变量。

    返回:
    list: OpenAI API的message格式列表
    """

    def format_grid(grid):
        """辅助函数：将网格格式化为字符串，每行一个列表。"""
        return "\n".join([str(row) for row in grid])

    # 1. 系统提示词：设定模型的角色和任务
    system_prompt = """你是一个解决抽象推理谜题的专家。
                        擅长于发现网格之间的变换关系
                        你的目标是分析训练样本（输入和输出网格），推导出隐藏的逻辑规则，然后将这条规则应用到测试输入网格上。
                        所有网格都使用 0-9 的整数表示。
                        你必须严格按照用户要求的 JSON 格式进行思考和输出。
                        """

    # 2. 用户提示词：构建 CoT 指导
    user_prompt_lines = []
    user_prompt_lines.append(
        "你将收到一系列“输入-输出”对。所有网格都由整数 0-9 组成。")
    user_prompt_lines.append("\n--- 训练样本开始 ---")

    # 2.1. 格式化所有训练样本 (使用字符抽象)
    for i, example in enumerate(d['train']):
        input_grid = format_grid(example['input'])
        output_grid = format_grid(example['output'])  # 输出也用字符展示
        user_prompt_lines.append(f"\n样本 {i + 1} 输入:")
        user_prompt_lines.append(input_grid)
        user_prompt_lines.append(f"样本 {i + 1} 输出:")
        user_prompt_lines.append(output_grid)

    user_prompt_lines.append("\n--- 训练样本结束 ---")

    # 2.2. 提供测试输入 (使用字符抽象)
    user_prompt_lines.append("\n现在，这是需要你预测的测试输入网格：")
    user_prompt_lines.append(f"\n测试输入:")
    test_input_grid = format_grid(d['test'][0]['input'])
    user_prompt_lines.append(test_input_grid)

    # 2.3. 提供结构化 CoT (Structured CoT) 指导
    user_prompt_lines.append("\n--- 你的任务 ---")
    user_prompt_lines.append(
        "请分析所有样本，然后生成一个 *单一的 JSON 对象* 作为你的唯一回复。")
    user_prompt_lines.append("这个 JSON 对象必须严格遵循以下结构：")
    user_prompt_lines.append("""
{
  "reflection": "在此处详细反思你的观察。比较所有训练样本的输入和输出。网格的尺寸、形状、颜色（字符）或对象位置是如何变化的？",
  "overall_pattern": "在此处用一句话总结所有样本共有的最简洁的变换规则。",
  "final_output_grid": [[...]] 如[[0, 0], [0, 0]]
}
""")
    user_prompt_lines.append("\n--- 重要指令 ---")
    user_prompt_lines.append(
        "1. 你的整个回复 *必须* 只是一个从 `{` 开始到 `}` 结束的、格式正确的 JSON 对象。")
    user_prompt_lines.append(
        "2. **关键要求**: 在 `final_output_grid` 字段中，你必须使用原始的 **整数格式 (0-9)** 来表示网格。这是为了匹配评估标准。")


    user_content = "\n".join(user_prompt_lines)

    # 3. 组装成 OpenAI API 格式
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def parse_output(text):
    """
    解析大语言模型的输出文本，提取 "final_output_grid" 字段。
    (已更新以支持结构化 JSON CoT)

    参数:
    text (str): 大语言模型在设计prompt下的输出文本

    返回:
    list: 从输出文本解析出的二维数组 (Python列表，元素为整数)
          如果解析失败，返回空列表 []
    """
    try:
        # 1. 查找 JSON 对象。
        # re.DOTALL (re.S) 允许 '.' 匹配换行符
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            print("解析错误: 未找到 JSON 对象。") # 调试时使用
            return []

        json_string = match.group(0)

        # 2. 解析 JSON
        parsed_json = json.loads(json_string)

        # 3. 提取 "final_output_grid" 字段
        grid_list = parsed_json.get('final_output_grid')
        # print(grid_list)

        if not isinstance(grid_list, list):
            print("解析错误: 'final_output_grid' 字段不是列表。") # 调试时使用
            return []

        # 4. (关键) 验证并强制将所有元素转换为整数 (int)
        # 这是为了防止模型输出 "2" (字符串) 而不是 2 (整数)
        converted_list = []
        for r_idx, row in enumerate(grid_list):
            if not isinstance(row, list):
                # print(f"解析错误: 行 {r_idx} 不是列表。") # 调试时使用
                return []

            converted_row = []
            for c_idx, item in enumerate(row):
                try:
                    # 强制转换为 int
                    converted_row.append(int(item))
                except (ValueError, TypeError):
                    # 如果转换失败 (例如 "a" 或 null)
                    # print(f"解析警告: 无法将 (row {r_idx}, col {c_idx}) 的值 '{item}' 转换为整数。")
                    return []
            converted_list.append(converted_row)

        return converted_list

    except (json.JSONDecodeError, re.error, Exception) as e:
        # 捕获JSON解析错误、正则错误或任何其他异常
        print(f"解析时发生意外错误: {e}") # 调试时使用


    # 如果任何步骤失败，返回空列表
    return []

