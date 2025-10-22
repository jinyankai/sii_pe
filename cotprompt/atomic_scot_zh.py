import json
import re


def construct_prompt(d):
    """
    构造用于大语言模型的提示词 (结构化CoT中文版)。
    引导模型进行CoT思考并输出结构化的JSON。

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
你的目标是分析网格变换的训练样本，推导出隐藏的逻辑规则，然后将该规则精确地应用到一个新的测试输入网格上。
"""

    # 2. 用户提示词：构建 CoT 指导
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

    # 2.3. [!!] 提供更广泛的中文分析策略
    user_prompt_lines.append("\n--- 指导说明 ---")
    user_prompt_lines.append(
        "你的任务是根据训练样本推导出变换规则，并将该规则应用于“测试输入”以生成“测试输出”。")
    user_prompt_lines.append(
        "\n让我们一步一步思考来找出规则。从高层分析开始，然后逐步深入。")

    user_prompt_lines.append("\n**第 1 步：高层分析（宏观视角）**")
    user_prompt_lines.append(
        "   - **变换类型：** 这是一个元素级的修改、一个基于对象的变换，还是一个“分解-选择”任务？")
    user_prompt_lines.append(
        "   - **分解-选择检查：** 输出网格看起来像是输入网格*某一部分*（子网格、对象或块）的*精确副本*吗？（例如，输入是 9x3，输出是 3x3）。如果是，那么任务就是找到*选择规则*（例如，‘选择包含最多 5 的块’，‘选择总和最小的块’）。")
    user_prompt_lines.append(
        "   - **几何操作：** 是否存在全局变换？（例如，旋转90度、垂直翻转、镜像）。")
    user_prompt_lines.append(
        "   - **对称性与重复：** 输出是否补全了一个模式、消除了不对称性，或者是平铺了输入中的某个形状？")

    user_prompt_lines.append("\n**第 2 步：对象级分析（如果不是“分解-选择”任务）**")
    user_prompt_lines.append(
        "   - **对象识别：** 网格是否可以被分解为独立的对象（由相同数字/值构成的形状）？")
    user_prompt_lines.append(
        "   - **对象命运：** 每个对象发生了什么？（例如，移动、数字改变、缩放、扭曲、删除、复制）。")
    user_prompt_lines.append(
        "   - **对象间关系：** 变换是否依赖于对象之间的关系？（例如，‘将小形状移入大形状内部’，‘在两个 8 之间画一条线’）。")

    user_prompt_lines.append("\n**第 3 步：元素级分析（如果没有清晰的对象规则）**")
    user_prompt_lines.append(
        "   - **维度：** 输出维度 `(H_out, W_out)` 与输入 `(H_in, W_in)` 有何关系？（例如，相同、`2*H_in`、`H_in / 2`）。")
    user_prompt_lines.append(
        "   - **元素映射：** 输出位置 `(r, c)` 处的元素值是如何得到的？它是否来自输入中对应的 `(r, c)` 或 `(r/2, c/2)`？")
    user_prompt_lines.append(
        "   - **邻域逻辑：** 输出元素 `(r, c)` 的值是否取决于输入元素 `(r, c)` 的*邻居*？（例如，‘如果一个单元格有超过3个值为2的邻居，它就变为1’）。")
    user_prompt_lines.append(
        "   - **数字/值逻辑：** 数字/值是否被系统地改变了？（例如，‘所有的 2 都变成了 4’，‘出现最频繁的数字变成了 8’）。")

    user_prompt_lines.append("\n**第 4 步：形成并验证规则**")
    user_prompt_lines.append(
        "   - **形成假设：** 基于第 1-3 步，陈述一个简单、清晰的规则假设。")
    user_prompt_lines.append(
        "   - **验证假设：** **关键步骤。** 将你的规则假设应用于*每一个*训练样本。它是否能完美预测*所有*样本的输出？如果它在任何一个样本上失败，你的假设就是错误的。请丢弃它并寻找新规则。")

    user_prompt_lines.append("\n**第 5 步：应用最终规则**")
    user_prompt_lines.append(
        "   - 一旦你有了一个在所有训练样本上 100% 一致的规则，就将该确切规则应用于‘测试输入’以生成‘测试输出’。")

    # 2.4. 规定最终的输出格式 (Structured JSON)
    user_prompt_lines.append("\n**最终答案格式：**")
    user_prompt_lines.append(
        "你*必须*只提供一个单一的、有效的 JSON 对象作为你的最终答案。不要在 JSON 块之前或之后添加任何其他文本或解释。")
    user_prompt_lines.append("该 JSON 对象必须具有以下结构：")
    user_prompt_lines.append("""
{
  "analysis": "遵循上述步骤的简要分析。明确说明测试了哪些假设，以及保留或丢弃它们的原因。如果是一个选择任务，请列出每个子网格的属性。",
  "rule": "对最终验证过的变换规则的简洁陈述。",
  "predicted_grid": [[...]]
}
""")
    user_prompt_lines.append("所需输出格式示例（你的内容将会不同）：")
    user_prompt_lines.append("""
{
  "analysis": "分析了所有3个训练样本。输入和输出网格始终是 3x3。规则似乎是水平翻转。验证：样本1的输入 [1,2,3] 在输出中变为 [3,2,1]，匹配。已在所有样本上验证。",
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
