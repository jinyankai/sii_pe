import json
import re


def construct_prompt(d):
    """
    构造用于大语言模型的提示词 (结构化思维树 ToT 中文版)。
    引导模型生成多个假设，然后评估并选择最佳假设。

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

    def get_grid_dims(grid):
        """辅助函数：获取网格的高度和宽度。"""
        if not grid:
            return 0, 0
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        return height, width

    # 1. 系统提示词：设定模型的角色和任务
    system_prompt = """你是一位解决抽象视觉推理谜题的专家，特别是来自 ARC (Abstraction and Reasoning Corpus) 的谜题。
你将使用“思维树”(Tree of Thoughts)方法论来系统地探索、评估和解决问题。
你的目标是分析训练样本，推导出隐藏的逻辑规则，然后将该规则精确地应用到测试输入网格上。
"""

    # 2. 用户提示词：构建 ToT 指导
    user_prompt_lines = []
    user_prompt_lines.append("这里有几个训练样本。请你找出变换规则。")
    user_prompt_lines.append("\n--- 训练样本开始 ---")

    # 2.1. 格式化所有训练样本
    for i, example in enumerate(d['train']):
        input_grid = example['input']
        output_grid = example['output']

        in_h, in_w = get_grid_dims(input_grid)
        out_h, out_w = get_grid_dims(output_grid)

        user_prompt_lines.append(f"\n样本 {i + 1} 输入 (高度: {in_h}, 宽度: {in_w}):")
        user_prompt_lines.append(format_grid(input_grid))
        user_prompt_lines.append(f"样本 {i + 1} 输出 (高度: {out_h}, 宽度: {out_w}):")
        user_prompt_lines.append(format_grid(output_grid))

    user_prompt_lines.append("\n--- 训练样本结束 ---")

    # 2.2. 提供测试输入
    user_prompt_lines.append("\n现在，这是需要你预测的测试输入网格：")

    test_input_grid = d['test'][0]['input']
    test_h, test_w = get_grid_dims(test_input_grid)

    user_prompt_lines.append(f"\n测试输入 (高度: {test_h}, 宽度: {test_w}):")
    user_prompt_lines.append(format_grid(test_input_grid))

    # 2.3. [!! 已修改 !!] 切换到“思维树” (ToT) 指导
    user_prompt_lines.append("\n--- 指导说明 ---")
    user_prompt_lines.append(
        "你的任务是根据训练样本推导出变换规则，并将该规则应用于“测试输入”以生成“测试输出”。")
    user_prompt_lines.append(
        "\n让我们使用“思维树”（Tree of Thoughts）的方法来系统地找出规则。")

    user_prompt_lines.append("\n**第 1 步：分析与假设生成 (Analyze & Generate Hypotheses)**")
    user_prompt_lines.append(
        "   - 仔细观察所有训练样本。")
    user_prompt_lines.append(
        "   - **生成 2-3 个关于变换规则的独立假设 (Hypotheses)。**")
    user_prompt_lines.append(
        "   - 你的假设应该涵盖不同的可能性，例如：")
    user_prompt_lines.append(
        "       - *假设 1 (例如，分解-选择):* '输出是输入的某个子块。选择规则是...'")
    user_prompt_lines.append(
        "       - *假设 2 (例如，对象级):* '输入中的'红色对象'被'蓝色对象'包围...'")
    user_prompt_lines.append(
        "       - *假设 3 (例如，元素级):* '所有值为 2 的元素，如果其邻居有 8，则变为 1...'")

    user_prompt_lines.append("\n**第 2 步：假设评估与验证 (Evaluate & Verify Hypotheses)**")
    user_prompt_lines.append(
        "   - **逐个评估**你在第 1 步中生成的每个假设。")
    user_prompt_lines.append(
        "   - **关键步骤：** 将每个假设与*所有*训练样本进行严格对比。")
    user_prompt_lines.append(
        "   - 演示你的验证过程（内心思考，在最终分析中总结）：")
    user_prompt_lines.append(
        "       - *评估假设 1:* 样本 1 (成功/失败?), 样本 2 (成功/失败?), ...")
    user_prompt_lines.append(
        "       - *评估假设 2:* 样本 1 (成功/失败?), 样本 2 (成功/失败?), ...")
    user_prompt_lines.append(
        "   - **剪枝 (Pruning):** 任何在一个或多个样本上失败的假设都*必须*被丢弃。")

    user_prompt_lines.append("\n**第 3 步：选择与应用 (Select & Apply)**")
    user_prompt_lines.append(
        "   - 报告在第 2 步中唯一通过了*所有*训练样本验证的假设。")
    user_prompt_lines.append(
        "   - （如果所有假设都失败了，请返回第 1 步并生成新的假设进行分析。）")
    user_prompt_lines.append(
        "   - 将这个经过验证的规则应用于“测试输入”以生成“测试输出”。")


    # 2.4. 规定最终的输出格式 (Structured JSON)
    user_prompt_lines.append("\n**最终答案格式：**")
    user_prompt_lines.append(
        "你*必须*只提供一个单一的、有效的 JSON 对象作为你的最终答案。不要在 JSON 块之前或之后添加任何其他文本或解释。")
    user_prompt_lines.append("该 JSON 对象必须具有以下结构：")
    user_prompt_lines.append("""
{
  "analysis": "遵循上述ToT步骤的简要分析。明确说明你生成了哪些假设，以及它们是如何被验证或剪枝的。如果是一个选择任务，请列出每个子网格的属性。",
  "rule": "对最终被选中的、经过验证的变换规则的简洁陈述。",
  "predicted_grid": [[...]]
}
""")
    user_prompt_lines.append("所需输出格式示例（你的内容将会不同）：")
    user_prompt_lines.append("""
{
  "analysis": "我生成了三个假设。\n假设1 (最大和选择): 在样本2上失败。\n假设2 (水平翻转): 在所有样本上均成功。\n假设3 (颜色替换): 在样本1上失败。\n因此，我选择假设2。",
  "rule": "将输入网格水平翻转以产生输出网格。",
  "predicted_grid": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
}
""")

    user_content = "\n".join(user_prompt_lines)

    # 3. 组装成 OpenAI API 格式
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def parse_output(text):
    """
    解析大语言模型的输出文本，提取预测的网格。
    该版本假定模型会输出一个包含 "predicted_grid" 键的 JSON 对象。

    参数:
    text (str): 大语言模型在设计prompt下的输出文本

    返回:
    list: 从输出 JSON 对象中提取的 "predicted_grid" (二维数组)
          如果解析失败，返回空列表 []
    """
    try:
        # 使用正则表达式查找所有类似 JSON 对象的字符串
        # re.DOTALL (re.S) 允许 '.' 匹配换行符，以处理跨越多行的JSON
        # '.*?' 是非贪婪匹配
        matches = re.findall(r"(\{.*?\})", text, re.DOTALL)

        if matches:
            # 假设模型可能在最后才输出正确答案，因此我们取最后一个匹配项
            json_string = matches[-1]

            # 清理可能的 markdown 代码围栏
            json_string = json_string.strip().lstrip('```json').lstrip('```').rstrip('```')

            # 尝试解析这个字符串为Python字典
            parsed_json = json.loads(json_string)

            # 检查它是否是一个字典并且有我们需要的键
            if isinstance(parsed_json, dict):
                predicted_grid = parsed_json.get("predicted_grid")

                # 基本的类型验证
                if isinstance(predicted_grid, list):
                    # 确保它是一个列表的列表（或者是空列表）
                    if not predicted_grid or all(isinstance(row, list) for row in predicted_grid):
                        return predicted_grid

    except (json.JSONDecodeError, re.error, Exception) as e:
        # 捕获JSON解析错误、正则错误或任何其他异常
        # print(f"解析输出时出错: {e}") # 可以在调试时取消注释
        pass

    # 如果没有找到匹配项，或者解析失败，或者 "predicted_grid" 格式不正确
    # 尝试回退到原始 `atomic_cot.py` 的解析逻辑（只查找网格列表）
    try:
        matches = re.findall(r"(\[\[.*?\]\])", text, re.DOTALL)
        if matches:
            json_string = matches[-1]
            parsed_list = json.loads(json_string)
            if isinstance(parsed_list, list):
                if not parsed_list or all(isinstance(row, list) for row in parsed_list):
                    return parsed_list
    except Exception:
        pass  # 最终解析失败

    return []
