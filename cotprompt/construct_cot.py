import json
import re


def construct_prompt(d):
    """
    构造用于大语言模型的提示词 (思维链版本 - 中文)

    参数:
    d (dict): jsonl数据文件的一行，解析成字典后的变量。
              注意：传入的 'd' 已经过处理，其 'test' 字段列表
              只包含 'input'，不包含 'output' 答案。

    返回:
    list: OpenAI API的message格式列表，允许设计多轮对话式的prompt
    """

    def format_grid(grid):
        """辅助函数：将网格格式化为字符串，每行一个列表。"""
        return "\n".join([str(row) for row in grid])

    # 1. 系统提示词：设定模型的角色和任务
    system_prompt = """你是一个解决抽象视觉推理谜题的专家，特别是来自 ARC (Abstraction and Reasoning Corpus) 的谜题。
你的目标是分析几个网格变换的训练样本，推导出隐藏的逻辑规则，然后将这个规则精确地应用到一个新的测试输入网格上。
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

    # 2.3. 提供思维链 (CoT) 指导
    user_prompt_lines.append("\n--- 指导说明 ---")
    user_prompt_lines.append("你的任务是根据训练样本推导出变换规则，并将该规则应用于“测试输入”以生成“测试输出”。")
    user_prompt_lines.append("\n让我们一步一步思考来找出规则：")
    user_prompt_lines.append("1. **分析样本：** 查看所有的输入-输出对。它们之间一致的变化是什么？")
    user_prompt_lines.append("   - **尺寸：** 输出网格的尺寸是否与输入网格不同？")
    user_prompt_lines.append("   - **颜色（数字）：** 是否有颜色被添加、移除或改变？是否有些颜色保持不变？")
    user_prompt_lines.append("   - **几何操作：** 整个网格或其一部分是否被移动（平移）、旋转或翻转（水平、垂直或对角线）？")
    user_prompt_lines.append(
        "   - **模式匹配：** 是否在输入中识别了特定的模式或形状，然后对其进行修改、复制或用它来创建输出？")
    user_prompt_lines.append(
        "   - **元素分析：** 输入中特定位置 `(行, 列)` 的元素与输出中相同或不同位置的元素有何关系？这种变化是基于它的邻居还是它的绝对位置？")

    user_prompt_lines.append("\n2. **形成规则：** 总结出一个能解释 *所有* 训练样本的最简洁的规则。")

    user_prompt_lines.append("\n3. **应用规则：** 将这条规则应用到“测试输入”上，以产生“测试输出”。")

    # 2.4. 规定最终的输出格式
    user_prompt_lines.append("\n**最终答案格式：**")
    user_prompt_lines.append(
        "在你的推理分析之后，请 *仅仅* 提供最终预测的输出网格，格式为 2D 列表的 JSON 数组。在最终的网格之后不要添加任何其他文本或解释。")
    user_prompt_lines.append("最终答案格式示例：[[0, 1, 0], [1, 0, 1], [0, 1, 0]]")

    user_content = "\n".join(user_prompt_lines)

    # 3. 组装成 OpenAI API 格式
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def parse_output(text):
    """
    解析大语言模型的输出文本，提取预测的网格

    参数:
    text (str): 大语言模型在设计prompt下的输出文本

    返回:
    list: 从输出文本解析出的二维数组 (Python列表，元素为整数)
          如果解析失败，返回空列表 []
    """
    try:
        # 使用正则表达式查找所有类似2D列表的字符串
        # re.DOTALL (re.S) 允许 '.' 匹配换行符，以处理跨越多行的JSON数组
        # '.*?' 是非贪婪匹配
        matches = re.findall(r"(\[\[.*?\]\])", text, re.DOTALL)

        if matches:
            # 假设模型可能在最后才输出正确答案，因此我们取最后一个匹配项
            json_string = matches[-1]

            # 尝试解析这个字符串为Python列表
            parsed_list = json.loads(json_string)

            # 基本的类型验证
            if isinstance(parsed_list, list):
                # 确保它是一个列表的列表（或者是空列表）
                if not parsed_list or all(isinstance(row, list) for row in parsed_list):
                    return parsed_list

    except (json.JSONDecodeError, re.error, Exception):
        # 捕获JSON解析错误、正则错误或任何其他异常
        # print(f"Error parsing output: {e}") # 可以在调试时取消注释
        pass

    # 如果没有找到匹配项，或者解析失败，返回一个空列表
    return []

