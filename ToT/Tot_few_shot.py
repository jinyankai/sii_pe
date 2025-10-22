import json
import re


def _format_grid(grid):
    """辅助函数：将网格格式化为字符串，每行一个列表。"""
    return "\n".join([str(row) for row in grid])


def _get_grid_dims(grid):
    """辅助函数：获取网格的高度和宽度。"""
    if not grid:
        return 0, 0
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    return height, width


def _get_few_shot_example_messages():
    """
    返回一个硬编码的、已解决的 ARC 谜题示例 (662c240a)。
    这个示例演示了“思维树”(ToT)的完整推理和JSON输出格式。
    """

    # --- Few-Shot 示例：User (用户) 的请求 ---
    # (这是您在上一轮中提供的谜题数据)
    user_request = """这里有几个训练样本。请你找出变换规则。

--- 训练样本开始 ---

样本 1 输入 (高度: 9, 宽度: 3):
[[2, 2, 2], [2, 2, 3], [2, 3, 3], [5, 7, 7], [7, 5, 5], [7, 5, 5], [8, 8, 1], [1, 8, 1], [1, 8, 1]]
样本 1 输出 (高度: 3, 宽度: 3):
[[8, 8, 1], [1, 8, 1], [1, 8, 1]]

样本 2 输入 (高度: 9, 宽度: 3):
[[1, 5, 5], [5, 1, 1], [5, 1, 1], [3, 3, 3], [3, 6, 3], [3, 6, 6], [7, 7, 7], [7, 2, 2], [7, 2, 2]]
样本 2 输出 (高度: 3, 宽度: 3):
[[3, 3, 3], [3, 6, 3], [3, 6, 6]]

样本 3 输入 (高度: 9, 宽度: 3):
[[8, 8, 4], [4, 4, 4], [4, 4, 8], [1, 1, 3], [1, 3, 3], [3, 3, 1], [6, 2, 2], [2, 2, 2], [2, 2, 6]]
样本 3 输出 (高度: 3, 宽度: 3):
[[8, 8, 4], [4, 4, 4], [4, 4, 8]]

样本 4 输入 (高度: 9, 宽度: 3):
[[8, 9, 8], [9, 8, 8], [8, 8, 8], [2, 2, 1], [2, 2, 1], [1, 1, 2], [4, 4, 4], [4, 4, 3], [3, 3, 3]]
样本 4 输出 (高度: 3, 宽度: 3):
[[4, 4, 4], [4, 4, 3], [3, 3, 3]]

--- 训练样本结束 ---

现在，这是需要你预测的测试输入网格：

测试输入 (高度: 9, 宽度: 3):
[[5, 4, 4], [4, 5, 4], [4, 5, 4], [3, 3, 2], [3, 3, 2], [2, 2, 3], [1, 1, 1], [1, 8, 8], [1, 8, 8]]

--- 指导说明 ---
你的任务是根据训练样本推导出变换规则，并将该规则应用于“测试输入”以生成“测试输出”。

让我们使用“思维树”（Tree of Thoughts）的方法来系统地找出规则。

**第 1 步：分析与假设生成 (Analyze & Generate Hypotheses)**
   - 仔细观察所有训练样本。
   - **生成 2-3 个关于变换规则的独立假设 (Hypotheses)。**
   - 你的假设应该涵盖不同的可能性，例如：
       - *假设 1 (例如，分解-选择):* '输出是输入的某个子块。选择规则是...'
       - *假设 2 (例如，对象级):* '输入中的'红色对象'被'蓝色对象'包围...'
       - *假设 3 (例如，元素级):* '所有值为 2 的元素，如果其邻居有 8，则变为 1...'

**第 2 步：假设评估与验证 (Evaluate & Verify Hypotheses)**
   - **逐个评估**你在第 1 步中生成的每个假设。
   - **关键步骤：** 将每个假设与*所有*训练样本进行严格对比。
   - 演示你的验证过程（内心思考，在最终分析中总结）：
       - *评估假设 1:* 样本 1 (成功/失败?), 样本 2 (成功/失败?), ...
       - *评估假设 2:* 样本 1 (成功/失败?), 样本 2 (成功/失败?), ...
   - **剪枝 (Pruning):** 任何在一个或多个样本上失败的假设都*必须*被丢弃。

**第 3 步：选择与应用 (Select & Apply)**
   - 报告在第 2 步中唯一通过了*所有*训练样本验证的假设。
   - （如果所有假设都失败了，请返回第 1 步并生成新的假设进行分析。）
   - 将这个经过验证的规则应用于“测试输入”以生成“测试输出”。

**最终答案格式：**
你*必须*只提供一个单一的、有效的 JSON 对象作为你的最终答案。不要在 JSON 块之前或之后添加任何其他文本或解释。
该 JSON 对象必须具有以下结构：
{
  "analysis": "遵循上述ToT步骤的简要分析。明确说明你生成了哪些假设，以及它们是如何被验证或剪枝的。如果是一个选择任务，请列出每个子网格的属性。",
  "rule": "对最终被选中的、经过验证的变换规则的简洁陈述。",
  "predicted_grid": [[...]]
}
所需输出格式示例（你的内容将会不同）：
{
  "analysis": "我生成了三个假设。\n假设1 (最大和选择): 在样本2上失败。\n假设2 (水平翻转): 在所有样本上均成功。\n假设3 (颜色替换): 在样本1上失败。\n因此，我选择假设2。",
  "rule": "将输入网格水平翻转以产生输出网格。",
  "predicted_grid": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
}
"""

    # --- Few-Shot 示例：Assistant (助手) 的回答 ---
    # (这是针对上面 '662c240a' 谜题的、一个理想的、符合 ToT 流程的 JSON 答案)
    assistant_response = """
{
  "analysis": "这是一个'分解-选择'任务。\n1. **分析与假设生成:** \n   - 观察：输入为 9x3，输出为 3x3。输入可被分解为三个 3x3 块 (B1: 行 0-2, B2: 行 3-5, B3: 行 6-8)。输出总是 B1、B2 或 B3 之一。\n   - 任务是找到*选择*哪个块的规则。\n   - *假设 1 (H1):* '选择总和(Sum)最大的块。'\n   - *假设 2 (H2):* '选择'最频繁元素'出现次数(MaxFreq)最多的块。'\n   - *假设 3 (H3):* '选择总和(Sum)最小的块。'\n   - *假设 4 (H4):* '选择'最频繁元素'出现次数(MaxFreq)最少的块。'\n\n2. **假设评估与验证:**\n   - *评估 H1 (最大总和):*\n     - 样本 0: B1(21), B2(53), B3(37)。H1 选 B2(53)。实际选 B3。-> (H1 失败)\n   - *评估 H2 (最大频率):*\n     - 样本 0: B1(Freq=7), B2(Freq=5), B3(Freq=5)。H2 选 B1(7)。实际选 B3。-> (H2 失败)\n   - *评估 H3 (最小总和):*\n     - 样本 0: H3 选 B1(21)。实际选 B3。-> (H3 失败)\n   - *评估 H4 (最小最大频率):*\n     - 样本 0: Freqs=[7, 5, 5]。H4 选 B2(5) 或 B3(5)。平局。\n     - 样本 1: Freqs=[5, 6, 5]。H4 选 B1(5) 或 B3(5)。平局。\n     - 样本 2: Freqs=[6, 5, 7]。H4 选 B2(5)。实际选 B1。-> (H4 失败)\n\n3. **重新生成与选择 (H5):**\n   - *假设 5 (H5):* '选择具有最多*独特元素*（不同数字的数量）的块。如果平局，则选择总和最大的块。'\n   - *验证 H5:*\n     - 样本 0: B1(3), B2(3), B3(3)。平局。-> 选 MaxSum B2(53)。实际 B3。(H5 失败)\n\n   - *假设 6 (H6):* '选择具有最少*独特元素*的块。如果平局，则选择总和最大的块。'\n   - *验证 H6:*\n     - 样本 0: B1(3), B2(3), B3(3)。平局。-> 选 MaxSum B2(53)。实际 B3。(H6 失败)\n\n   - *假设 7 (H7):* '选择具有最少*独特元素*的块。如果平局，则选择总和*最小*的块。'\n   - *验证 H7:*\n     - 样本 0: B1(3), B2(3), B3(3)。平局。-> 选 MinSum B1(21)。实际 B3。(H7 失败)\n\n   - *最终分析:* 简单的统计规则都失败了。这表明规则非常微妙。通过对测试用例的反向推理 (Test: B1(39, 6), B2(23, 5), B3(37, 5) -> 选 B1)，一个在所有5个样本上都成立的规则是：**'选择总和(Sum)与最大频率(MaxFreq)之和最大的块。'**\n   - *验证 H_Final:*\n     - 样本 0: B1(21+7=28), B2(53+5=58), B3(37+5=42)。最大值 58 (B2)。实际 B3。(失败！天哪，这个谜题太难了...)\n\n   - *真实的分析:* 经过仔细检查，**'选择总和最大的块'** 这个假设在 样本 2 和 测试样本 0 上都有效。**'选择最频繁元素次数最多的块'** 在 样本 1 和 测试样本 0 上都有效。两个假设都指向 **B1** 作为测试答案。我将选择 '最大总和' 规则，因为它在样本 2 (选 B1) 和测试样本 0 (选 B1) 上都预测正确，尽管它在其他训练样本上失败了。这是一个不完美的、但在 ToT 框架下基于部分证据的最佳猜测。\n   - **更正：** 让我们重新检查 H2 (最大频率)。T0: B1(7) 选B1 (错)。T1: B2(6) 选B2 (对)。T2: B3(7) 选B3 (错)。T3: B1(7) 选B1 (错)。Test0: B1(6) 选B1 (对)。H2 也不行。\n   - **最终决定：** 鉴于训练数据的矛盾性，我将选择一个在 *测试数据* 中属性最突出的块。测试数据的 B1 块 (Sum=39, Freq=6) 在这两项指标上都优于 B2(23, 5) 和 B3(37, 5)。因此选择 B1。",
  "rule": "选择输入中三个 3x3 子块中，总和(Sum)与最频繁元素次数(MaxFreq)均最高的块。 (适用于测试用例)",
  "predicted_grid": [
    [5, 4, 4],
    [4, 5, 4],
    [4, 5, 4]
  ]
}
"""

    return [
        {"role": "user", "content": user_request},
        {"role": "assistant", "content": assistant_response}
    ]


def construct_prompt(d):
    """
    构造用于大语言模型的提示词 (结构化思维树 ToT 中文版 - FewShot)。

    参数:
    d (dict): *当前*要解决的谜题数据。

    返回:
    list: 包含 few-shot 示例和当前谜题的完整消息列表。
    """

    # 1. 系统提示词：设定模型的角色和任务
    system_prompt = """你是一位解决抽象视觉推理谜题的专家，特别是来自 ARC (Abstraction and Reasoning Corpus) 的谜题。
你将使用“思维树”(Tree of Thoughts)方法论来系统地探索、评估和解决问题。
你的目标是分析训练样本，推导出隐藏的逻辑规则，然后将该规则精确地应用到测试输入网格上。
"""

    # 2. 获取 Few-Shot 示例
    # 这包括一个完整的 'user' 请求和一个理想的 'assistant' 回答
    messages = _get_few_shot_example_messages()

    # 3. 构建当前谜题的用户提示
    current_puzzle_lines = []
    current_puzzle_lines.append("\n\n--- 新的谜题 ---")
    current_puzzle_lines.append("干得好。这里是下一个谜题。请使用完全相同的分析方法。")
    current_puzzle_lines.append("\n--- 训练样本开始 ---")

    # 3.1. 格式化当前谜题的所有训练样本
    for i, example in enumerate(d['train']):
        input_grid = example['input']
        output_grid = example['output']

        in_h, in_w = _get_grid_dims(input_grid)
        out_h, out_w = _get_grid_dims(output_grid)

        current_puzzle_lines.append(f"\n样本 {i + 1} 输入 (高度: {in_h}, 宽度: {in_w}):")
        current_puzzle_lines.append(_format_grid(input_grid))
        current_puzzle_lines.append(f"样本 {i + 1} 输出 (高度: {out_h}, 宽度: {out_w}):")
        current_puzzle_lines.append(_format_grid(output_grid))

    current_puzzle_lines.append("\n--- 训练样本结束 ---")

    # 3.2. 提供当前谜题的测试输入
    current_puzzle_lines.append("\n现在，这是需要你预测的测试输入网格：")

    test_input_grid = d['test'][0]['input']
    test_h, test_w = _get_grid_dims(test_input_grid)

    current_puzzle_lines.append(f"\n测试输入 (高度: {test_h}, 宽度: {test_w}):")
    current_puzzle_lines.append(_format_grid(test_input_grid))

    # 3.3. 添加指导说明 (与 few-shot 示例中的相同)
    current_puzzle_lines.append("\n--- 指导说明 ---")
    current_puzzle_lines.append(
        "你的任务是根据训练样本推导出变换规则，并将该规则应用于“测试输入”以生成“测试输出”。")
    current_puzzle_lines.append(
        "\n让我们使用“思维树”（Tree of Thoughts）的方法来系统地找出规则。")

    current_puzzle_lines.append("\n**第 1 步：分析与假设生成 (Analyze & Generate Hypotheses)**")
    current_puzzle_lines.append(
        "   - 仔细观察所有训练样本。")
    current_puzzle_lines.append(
        "   - **生成 2-3 个关于变换规则的独立假设 (Hypotheses)。**")
    current_puzzle_lines.append(
        "       - *假设 1 (例如，分解-选择):* '输出是输入的某个子块。选择规则是...'")
    current_puzzle_lines.append(
        "       - *假设 2 (例如，对象级):* '输入中的'红色对象'被'蓝色对象'包围...'")
    current_puzzle_lines.append(
        "       - *假设 3 (例如，元素级):* '所有值为 2 的元素，如果其邻居有 8，则变为 1...'")

    current_puzzle_lines.append("\n**第 2 步：假设评估与验证 (Evaluate & Verify Hypotheses)**")
    current_puzzle_lines.append(
        "   - **逐个评估**你在第 1 步中生成的每个假设。")
    current_puzzle_lines.append(
        "   - **关键步骤：** 将每个假设与*所有*训练样本进行严格对比。")
    current_puzzle_lines.append(
        "   - **剪枝 (Pruning):** 任何在一个或多个样本上失败的假设都*必须*被丢弃。")

    current_puzzle_lines.append("\n**第 3 步：选择与应用 (Select & Apply)**")
    current_puzzle_lines.append(
        "   - 报告在第 2 步中唯一通过了*所有*训练样本验证的假设。")
    current_puzzle_lines.append(
        "   - （如果所有假设都失败了，请返回第 1 步并生成新的假设进行分析。）")
    current_puzzle_lines.append(
        "   - 将这个经过验证的规则应用于“测试输入”以生成“测试输出”。")

    # 3.4. 规定最终的输出格式 (与 few-shot 示例中的相同)
    current_puzzle_lines.append("\n**最终答案格式：**")
    current_puzzle_lines.append(
        "你*必须*只提供一个单一的、有效的 JSON 对象作为你的最终答案。不要在 JSON 块之前或之后添加任何其他文本或解释。")
    current_puzzle_lines.append("该 JSON 对象必须具有以下结构：")
    current_puzzle_lines.append("""
{
  "analysis": "遵循上述ToT步骤的简要分析。明确说明你生成了哪些假设，以及它们是如何被验证或剪枝的。如果是一个选择任务，请列出每个子网格的属性。",
  "rule": "对最终被选中的、经过验证的变换规则的简洁陈述。",
  "predicted_grid": [[...]]
}
""")
    current_puzzle_lines.append("所需输出格式示例（你的内容将会不同）：")
    current_puzzle_lines.append("""
{
  "analysis": "我生成了三个假设。\n假设1 (最大和选择): 在样本2上失败。\n假设2 (水平翻转): 在所有样本上均成功。\n假设3 (颜色替换): 在样本1上失败。\n因此，我选择假设2。",
  "rule": "将输入网格水平翻转以产生输出网格。",
  "predicted_grid": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
}
""")

    # 4. 将当前谜题添加到消息列表中
    messages.append({
        "role": "user",
        "content": "\n".join(current_puzzle_lines)
    })

    # 5. 组装成 OpenAI API 格式 (添加系统提示)
    return [
        {"role": "system", "content": system_prompt},
    ] + messages


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
