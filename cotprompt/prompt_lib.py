import json
import re
import ast  # 用于安全解析 Python 字典字符串


# --- 辅助函数：格式化网格 ---

def _get_grid_dims(grid):
    """辅助函数：获取网格的高度和宽度。"""
    # ... (此部分辅助函数与上一版相同，保持不变) ...
    if not grid:
        return 0, 0
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    return height, width


def _format_grid_json(grid):
    """辅助函数：将网格格式化为 JSON 字符串。"""
    return json.dumps(grid)


def _format_grid_text(grid):
    """辅助函数：将网格格式化为纯文本，用空格分隔。"""
    return "\n".join([" ".join([str(cell) for cell in row]) for row in grid])


def _format_grid_pgm(grid, i, is_input):
    """辅助函数：将网格格式化为 PGM 字符串。"""
    label = "input" if is_input else "output"
    transform_id = chr(ord('A') + i)  # A, B, C...

    height, width = _get_grid_dims(grid)

    # PGM 头部: P2, 宽度 高度, 最大灰度值 (我们这里用 9)
    header = f"P2\n# {label}\n{width} {height}\n9"

    # PGM 数据
    data = _format_grid_text(grid)

    return f"## Transformation {transform_id} - {label.capitalize()}\n\n```pgm\n{header}\n{data}\n```"


def _format_grid_python_dict(grid, i, is_input):
    """辅助函数：将网格格式化为稀疏 Python 字典字符串。"""
    label = "input" if is_input else "output"
    height, width = _get_grid_dims(grid)

    # 构建稀疏字典
    grid_dict = {
        'width': width,
        'height': height,
        'background': 0  # 假设背景总是 0
    }

    for r in range(height):
        for c in range(width):
            if grid[r][c] != 0:
                grid_dict[(r, c)] = grid[r][c]

    # 格式化为字符串
    # 为了与仓库中的示例 (e.g., a699fb00-1.md) 保持一致，我们将使用 repr()
    # 它会正确处理元组键
    dict_str = repr(grid_dict)

    return f"{label}[{i}] = {dict_str}"


# --- 辅助函数：解析输出 ---

def _parse_grid_from_json_string(text):
    """
    辅助函数：从文本中稳健地提取最后一个 JSON 2D 列表。
    例如 '... [[0, 1], [1, 0]] ...'
    """
    # ... (此部分辅助函数与上一版相同，保持不变) ...
    try:
        # 查找所有类似 2D 列表的字符串
        # re.DOTALL (re.S) 允许 '.' 匹配换行符
        matches = re.findall(r"(\[\[.*?\]\])", text, re.DOTALL)
        if matches:
            # 取最后一个匹配项
            json_string = matches[-1]
            # 尝试解析
            grid = json.loads(json_string)
            if isinstance(grid, list) and (not grid or all(isinstance(row, list) for row in grid)):
                return grid
    except Exception:
        pass  # 解析失败
    return []


def _parse_grid_from_text(text_block):
    """
    辅助函数：从一个纯文本块中解析网格。
    例如 '0 1 0\n1 0 1\n0 1 0'
    """
    # ... (此部分辅助函数与上一版相同，保持不变) ...
    try:
        grid = []
        lines = text_block.strip().split('\n')
        if not lines:
            return []

        for line in lines:
            # 移除行首/行尾的方括号（如果有的话）
            clean_line = line.strip().strip('[]')
            if not clean_line:
                continue
            # 按空格或逗号分隔
            row = [int(x) for x in re.split(r'[,\s]+', clean_line) if x]
            grid.append(row)

        # 验证所有行的长度是否一致
        if grid and all(len(row) == len(grid[0]) for row in grid):
            return grid
    except Exception:
        pass
    return []


# --- 方法 1: 结构化 CoT ---
def _construct_prompt_structured_cot(d):
    """
    (私有) 构造用于大语言模型的提示词 (结构化 CoT 版本)。
    """

    def get_grid_dims_str(grid):
        # ... (此函数实现与上一版 construct_prompt_structured_cot 中的相同) ...
        h, w = _get_grid_dims(grid)
        return f"(Height: {h}, Width: {w})"

    # 1. 系统提示词
    system_prompt = """You are an expert at solving abstract visual reasoning puzzles, specifically those from the Abstraction and Reasoning Corpus (ARC).
Your goal is to analyze training examples of grid transformations, deduce the hidden logical rule, and then apply that rule precisely to a new test input grid.
"""

    # 2. 用户提示词
    user_prompt_lines = []
    user_prompt_lines.append("Here are several training examples. Please find the transformation rule.")
    user_prompt_lines.append("\n--- Training Examples Start ---")

    # 2.1. 格式化所有训练样本
    for i, example in enumerate(d['train']):
        input_grid_str = _format_grid_text(example['input'])
        output_grid_str = _format_grid_text(example['output'])

        user_prompt_lines.append(f"\nExample {i + 1} Input {get_grid_dims_str(example['input'])}:")
        user_prompt_lines.append(input_grid_str)
        user_prompt_lines.append(f"Example {i + 1} Output {get_grid_dims_str(example['output'])}:")
        user_prompt_lines.append(output_grid_str)

    user_prompt_lines.append("\n--- Training Examples End ---")

    # 2.2. 提供测试输入
    user_prompt_lines.append("\nNow, here is the test input grid you need to predict:")
    for i, test_case in enumerate(d['test']):
        test_input_grid_str = _format_grid_text(test_case['input'])
        user_prompt_lines.append(f"\nTest Input {i + 1} {get_grid_dims_str(test_case['input'])}:")
        user_prompt_lines.append(test_input_grid_str)

    # 2.3. 提供思维链 (CoT) 指导
    user_prompt_lines.append("\n--- Instructions ---")
    user_prompt_lines.append(
        "Your task is to deduce the transformation rule from the training examples and apply it to the 'Test Input' to generate the 'Test Output'.")
    user_prompt_lines.append("\nLet's think step-by-step to find the rule:")
    user_prompt_lines.append(
        "1. **Analyze Samples:** Look at all input-output pairs. What changes consistently? (Dimensions, colors/numbers, objects, geometric operations, pattern matching, element analysis).")
    user_prompt_lines.append(
        "2. **Formulate Rule:** Summarize the simplest, most concise rule that explains *all* training examples.")
    user_prompt_lines.append(
        "3. **Verify Rule:** Critically, test your rule. Mentally apply it to *each* training input. Does it perfectly produce the corresponding training output? If not, your rule is wrong. Refine it.")
    user_prompt_lines.append(
        "4. **Apply Rule:** Once verified, apply this exact rule to the 'Test Input' to determine the 'Test Output'.")

    # 2.4. 规定最终的输出格式 (结构化 JSON)
    user_prompt_lines.append("\n**Final Answer Format:**")
    user_prompt_lines.append(
        "You must provide *only* a single, valid JSON object as your final answer. Do not add any text or explanation before or after the JSON block.")
    user_prompt_lines.append("The JSON object must have the following structure:")
    user_prompt_lines.append("""
{
  "analysis": "A brief, step-by-step analysis of the training examples, identifying the core logic and how you derived the rule.",
  "rule": "A concise, one-sentence statement of the final transformation rule.",
  "predicted_grid": [[...]]
}
""")

    user_content = "\n".join(user_prompt_lines)

    # 3. 组装
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def _parse_output_structured_cot(text):
    """
    (私有) 解析结构化 CoT 版本的 LLM 输出，提取 "predicted_grid"。
    """
    # ... (此函数实现与上一版 parse_output_structured_cot 中的相同) ...
    try:
        # 查找 JSON 对象
        matches = re.findall(r"(\{.*?\})", text, re.DOTALL)
        if matches:
            # 取最后一个 JSON 对象
            json_string = matches[-1]
            # 清理可能的 markdown 标记
            json_string = json_string.strip().lstrip('```json').lstrip('```').rstrip('```')

            parsed_json = json.loads(json_string)

            if isinstance(parsed_json, dict):
                grid = parsed_json.get("predicted_grid")
                # 验证
                if isinstance(grid, list) and (not grid or all(isinstance(row, list) for row in grid)):
                    return grid

    except Exception:
        pass

    # 如果 JSON 解析失败，作为备用方案，尝试直接查找网格
    return _parse_grid_from_json_string(text)


# --- 方法 2: PGM 格式 + 多任务 ---
def _construct_prompt_pgm(d):
    """
    (私有) 构造用于大语言模型的提示词 (PGM 格式 + 多任务版本)。
    """

    # 1. 系统提示词
    # ... (此函数实现与上一版 construct_prompt_pgm 中的相同) ...
    system_prompt = "You solve puzzles by transforming the input into the output. You are expert at the PGM (Portable Graymap) format."

    # 2. 用户提示词
    user_prompt_lines = []
    user_prompt_lines.append("# Transformations")
    user_prompt_lines.append("Look for rules behind these transformations.\n")

    # 2.1. 格式化所有训练样本
    for i, example in enumerate(d['train']):
        user_prompt_lines.append(_format_grid_pgm(example['input'], i, is_input=True))
        user_prompt_lines.append(_format_grid_pgm(example['output'], i, is_input=False))

    # 2.2. 提供测试输入
    for i, test_case in enumerate(d['test']):
        # PGM 示例中的 Transform ID 通常是连续的
        transform_id = len(d['train']) + i
        user_prompt_lines.append(_format_grid_pgm(test_case['input'], transform_id, is_input=True))
        user_prompt_lines.append(f"## Transformation {chr(ord('A') + transform_id)} - Output\n")
        user_prompt_lines.append("```pgm\nP2\n# predict this output\n```")

    # 2.3. 多任务指导
    user_prompt_lines.append("\n# Task A: Analysis")
    user_prompt_lines.append("Think step by step, what does the transformation have in common. Use max 100 words.")
    user_prompt_lines.append("\n# Task B: Solution")
    user_prompt_lines.append(f"I want you to solve `Transformation {chr(ord('A') + len(d['train']))} - Output`.")
    user_prompt_lines.append("\n# Task C: Verify")
    user_prompt_lines.append("Double check that your reasoning is correct and check the solution. Use max 100 words.")

    user_content = "\n".join(user_prompt_lines)

    # 3. 组装
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def _parse_output_pgm(text):
    """
    (私有) 解析 PGM 版本的 LLM 输出，提取预测的网格。
    """
    # ... (此函数实现与上一版 parse_output_pgm 中的相同) ...
    try:
        # 查找最后一个 PGM 块
        matches = re.findall(r"```pgm\s*P2\s*#.*?output\s*([\s\S]*?)```", text, re.IGNORECASE)
        if matches:
            pgm_content = matches[-1].strip()

            # 分割头部和数据
            lines = pgm_content.split('\n')

            # 移除 P2, # comment, 和可能的空行
            data_lines = [line for line in lines if
                          line.strip() and not line.startswith('P2') and not line.startswith('#')]

            # 头部通常是 "width height" 和 "maxval"
            # 但在仓库的示例中 (e.g., land_rev_08_45_26_37_19.md)，头部是 "2 1\n9"
            # 而数据是 "7 3"
            # 在其他示例中 (land_ccw_01_38_62.md)，数据是 "1\n0"
            # 我们假设网格数据在 "9" (maxval) 之后开始

            grid_data_start_index = -1
            for i, line in enumerate(data_lines):
                if re.fullmatch(r'\d+', line.strip()):  # 找到像 "9" 这样的 maxval
                    grid_data_start_index = i + 1
                    break

            if grid_data_start_index != -1:
                grid_text_block = "\n".join(data_lines[grid_data_start_index:])
                return _parse_grid_from_text(grid_text_block)

    except Exception:
        pass

    # 备用方案：如果 PGM 解析失败，就查找最后一个看起来像网格的文本块
    if text.strip():
        last_block = text.split('\n\n')[-1]
        grid = _parse_grid_from_text(last_block)
        if grid:
            return grid

    return []


# --- 方法 3: Python 字典 + 多任务 ---
def _construct_prompt_python_dict(d):
    """
    (私有) 构造用于大语言模型的提示词 (Python 字典 + 多任务版本)。
    """

    # 1. 系统提示词
    # ... (此函数实现与上一版 construct_prompt_python_dict 中的相同) ...
    system_prompt = "I'm doing Python experiments.\nThese are images."

    # 2. 用户提示词
    user_prompt_lines = []
    user_prompt_lines.append("input = {}")
    user_prompt_lines.append("output = {}")

    # 2.1. 格式化所有训练样本
    for i, example in enumerate(d['train']):
        user_prompt_lines.append(_format_grid_python_dict(example['input'], i, is_input=True))
        user_prompt_lines.append(_format_grid_python_dict(example['output'], i, is_input=False))

    # 2.2. 提供测试输入
    test_index = len(d['train'])
    for i, test_case in enumerate(d['test']):
        user_prompt_lines.append(_format_grid_python_dict(test_case['input'], test_index + i, is_input=True))

    # 2.3. 多任务指导 (来自 e.g., a699fb00-1.md)
    user_prompt_lines.append("\n# Task A")
    user_prompt_lines.append(
        "Use at most 50 words.\nThink step by step.\n- Write notes about what shapes and patterns you observe.")

    user_prompt_lines.append("\n# Task B")
    user_prompt_lines.append(
        "Use at most 300 words.\nInclude a markdown formatted table with the most important observations about input and output images.")
    user_prompt_lines.append(
        "The table has three columns: observation name, observation values, comments about the observation.")
    user_prompt_lines.append(
        "Think step by step.\n- Count the mass of each layer.\n- Count how many strongly connected clusters...\n- Are there filled rectangles.\n- ... (etc.)")

    user_prompt_lines.append("\n# Task C")
    user_prompt_lines.append(
        "Use at most 100 words.\nThink step by step.\nWhat are the actions that converts input to output.")

    user_prompt_lines.append("\n# Task D")
    user_prompt_lines.append("With the following example input, I want you to predict what the output should be.")
    user_prompt_lines.append(
        f"Fill your predictions into the following template and replace PREDICT with your predictions.\n```python\noutput[{test_index}] = PREDICT\n```")

    user_content = "\n".join(user_prompt_lines)

    # 3. 组装
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def _parse_output_python_dict(text):
    """
    (私有) 解析 Python 字典版本的 LLM 输出，提取预测的网格。
    """
    # ... (此函数实现与上一版 parse_output_python_dict 中的相同) ...
    try:
        # 查找最后一个 'output[N] = PREDICT' 块
        matches = re.findall(r"output\[\d+\] = (\{.*?\})", text, re.DOTALL)
        if matches:
            dict_string = matches[-1]

            # 安全解析字符串为字典
            grid_dict = ast.literal_eval(dict_string)

            if isinstance(grid_dict, dict) and 'width' in grid_dict and 'height' in grid_dict:
                width = grid_dict['width']
                height = grid_dict['height']
                background = grid_dict.get('background', 0)

                # 创建一个填充了背景色的网格
                grid = [[background for _ in range(width)] for _ in range(height)]

                # 填充稀疏坐标
                for key, value in grid_dict.items():
                    if isinstance(key, tuple) and len(key) == 2:
                        r, c = key
                        if 0 <= r < height and 0 <= c < width:
                            grid[r][c] = value
                return grid

    except Exception:
        pass

    return []


# --- 方法 4: 自我修正 ---
def _construct_prompt_self_correct(d, maybe_output_grid):
    """
    (私有) 构造用于大语言模型的提示词 (自我修正版本)。
    """

    # 1. 系统提示词
    # ... (此函数实现与上一版 construct_prompt_self_correct 中的相同) ...
    system_prompt = "You are an ARC solver. Figure out whats wrong with the 'maybe output'. Then explain your reasoning and provide your own final answer. The answer must be different than the 'maybe output'."

    # 2. 用户提示词
    user_prompt_lines = []
    user_prompt_lines.append(
        "Find the common rule that maps an input grid to an output grid, given the examples below.")

    # 2.1. 格式化所有训练样本
    for i, example in enumerate(d['train']):
        user_prompt_lines.append(f"\nExample {i + 1}:\n")
        user_prompt_lines.append("Input:")
        user_prompt_lines.append(_format_grid_text(example['input']))
        user_prompt_lines.append("Output:")
        user_prompt_lines.append(_format_grid_text(example['output']))

    # 2.2. 提供测试输入和“可能的”答案
    user_prompt_lines.append(
        "\nBelow is a test input grid. Predict the corresponding output grid by applying the rule you found. Your final answer should just be the text output grid itself.")

    for i, test_case in enumerate(d['test']):
        user_prompt_lines.append("\nInput:")
        user_prompt_lines.append(_format_grid_text(test_case['input']))

    # 提供“可能的”错误答案
    user_prompt_lines.append("Maybe output:")
    user_prompt_lines.append(_format_grid_text(maybe_output_grid))

    user_content = "\n".join(user_prompt_lines)

    # 3. 组装
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def _parse_output_self_correct(text):
    """
    (私有) 解析自我修正版本的 LLM 输出。
    """
    # ... (此函数实现与上一版 parse_output_self_correct 中的相同) ...
    try:
        # 按空行分割，最后一个块很可能是网格
        blocks = text.strip().split('\n\n')
        if blocks:
            last_block = blocks[-1]
            grid = _parse_grid_from_text(last_block)
            if grid:
                return grid

        # 备用方案：如果不是用 \n\n 分隔，而是用单个换行
        lines = text.strip().split('\n')
        if not lines:
            return []

        # 从后往前找，看能组成网格的最长块是哪个
        for i in range(len(lines) - 1, -1, -1):
            block_to_test = "\n".join(lines[i:])
            grid = _parse_grid_from_text(block_to_test)
            if grid:
                return grid  # 找到的第一个有效网格

    except Exception:
        pass

    return []


# --- 公共调用接口 ---

def construct_prompt(d, strategy='structured_cot', **kwargs):
    """
    根据所选策略构造 ARC 提示。

    参数:
    d (dict): 任务字典。
    strategy (str): 'structured_cot', 'pgm', 'python_dict', 'self_correct' 之一。
    **kwargs:
        - maybe_output_grid (list): 'self_correct' 策略需要。

    返回:
    list: OpenAI API的message格式列表
    """
    if strategy == 'structured_cot':
        return _construct_prompt_structured_cot(d)
    elif strategy == 'pgm':
        return _construct_prompt_pgm(d)
    elif strategy == 'python_dict':
        return _construct_prompt_python_dict(d)
    elif strategy == 'self_correct':
        maybe_output = kwargs.get('maybe_output_grid')
        if not maybe_output:
            raise ValueError("strategy 'self_correct' requires 'maybe_output_grid' in kwargs.")
        return _construct_prompt_self_correct(d, maybe_output)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def parse_output(text, strategy='structured_cot'):
    """
    根据所选策略解析 LLM 的输出。

    参数:
    text (str): LLM 的原始输出文本。
    strategy (str): 'structured_cot', 'pgm', 'python_dict', 'self_correct' 之一。

    返回:
    list: 解析后的 2D 网格。
    """
    if strategy == 'structured_cot':
        return _parse_output_structured_cot(text)
    elif strategy == 'pgm':
        return _parse_output_pgm(text)
    elif strategy == 'python_dict':
        return _parse_output_python_dict(text)
    elif strategy == 'self_correct':
        return _parse_output_self_correct(text)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# --- 示例用法 ---
if __name__ == "__main__":
    # 这是一个示例 ARC 任务数据 (来自 25ff71a9)
    sample_task_data = {
        "train": [
            {"input": [[1, 1, 1], [0, 0, 0], [0, 0, 0]], "output": [[0, 0, 0], [1, 1, 1], [0, 0, 0]]},
            {"input": [[0, 0, 0], [1, 1, 1], [0, 0, 0]], "output": [[0, 0, 0], [0, 0, 0], [1, 1, 1]]}
        ],
        "test": [
            {"input": [[0, 1, 0], [1, 1, 0], [0, 0, 0]]}
            # "output" 字段在测试集中被省略
        ]
    }

    # --- 示例 1: 结构化 CoT ---
    print("--- 1. 'structured_cot' 策略 ---")
    prompt_messages = construct_prompt(sample_task_data, strategy='structured_cot')
    # print(json.dumps(prompt_messages, indent=2))
    print(f"System: {prompt_messages[0]['content']}")
    print(f"User: \n{prompt_messages[1]['content']}\n")

    # 模拟的 LLM 回复
    mock_response_cot = """
    Here is my analysis.

    ```json
    {
      "analysis": "The rule is to move the entire shape down by one row. If it reaches the bottom, it wraps around.",
      "rule": "Move the shape down by one row.",
      "predicted_grid": [[0, 0, 0], [0, 1, 0], [1, 1, 0]]
    }
    ```
    """
    print(f"--- 模拟回复 (Structured CoT) ---\n{mock_response_cot}")
    parsed_grid = parse_output(mock_response_cot, strategy='structured_cot')
    print(f"--- 解析结果 --- \n{parsed_grid}\n")

    # --- 示例 2: PGM 格式 ---
    print("--- 2. 'pgm' 策略 ---")
    prompt_messages_pgm = construct_prompt(sample_task_data, strategy='pgm')
    print(f"System: {prompt_messages_pgm[0]['content']}")
    print(f"User: \n{prompt_messages_pgm[1]['content']}\n")

    # 模拟的 LLM 回复
    mock_response_pgm = """
    Task A: The transformation moves the shape down by one row.
    Task B: Solution is below.
    Task C: The rule is verified.

    ```pgm
    P2
    # predict this output
    3 3
    9
    0 0 0
    0 1 0
    1 1 0
    ```
    """
    print(f"--- 模拟回复 (PGM) ---\n{mock_response_pgm}")
    parsed_grid_pgm = parse_output(mock_response_pgm, strategy='pgm')
    print(f"--- 解析结果 --- \n{parsed_grid_pgm}\n")

    # --- 示例 3: Python 字典 ---
    print("--- 3. 'python_dict' 策略 ---")
    prompt_messages_dict = construct_prompt(sample_task_data, strategy='python_dict')
    print(f"System: {prompt_messages_dict[0]['content']}")
    print(f"User: \n{prompt_messages_dict[1]['content']}\n")

    # 模拟的 LLM 回复
    mock_response_dict = """
    # Task A
    The shape moves down one row.

    # Task B
    | observation name | observation values | comments |
    |---|---|---|
    | Shape Movement | Yes, Down | Lines move one row down. |

    # Task C
    The action is to translate all non-background pixels down by 1 unit.

    # Task D
    ```python
    output[2] = {'width': 3, 'height': 3, 'background': 0, (1, 1): 1, (2, 0): 1, (2, 1): 1}
    ```
    """
    print(f"--- 模拟回复 (Python Dict) ---\n{mock_response_dict}")
    parsed_grid_dict = parse_output(mock_response_dict, strategy='python_dict')
    print(f"--- 解析结果 --- \n{parsed_grid_dict}\n")

    # --- 示例 4: 自我修正 ---
    print("--- 4. 'self_correct' 策略 ---")
    # 我们需要一个“可能的错误答案”
    bad_output = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # 假设 LLM 猜错了，以为是删除

    prompt_messages_correct = construct_prompt(sample_task_data, strategy='self_correct', maybe_output_grid=bad_output)
    print(f"System: {prompt_messages_correct[0]['content']}")
    print(f"User: \n{prompt_messages_correct[1]['content']}\n")

    # 模拟的 LLM 回复
    mock_response_correct = """
    The 'maybe output' is incorrect. It suggests the shape is deleted, but the training examples show the shape is *moved*.

    In Example 1, the line `1 1 1` at row 0 moves to row 1.
    In Example 2, the line `1 1 1` at row 1 moves to row 2.

    The rule is to move all non-zero pixels down by one row.

    Applying this to the test input `[[0, 1, 0], [1, 1, 0], [0, 0, 0]]`:
    - The `1` at (0, 1) moves to (1, 1).
    - The `1` at (1, 0) moves to (2, 0).
    - The `1` at (1, 1) moves to (2, 1).

    The correct output is:

    0 0 0
    0 1 0
    1 1 0
    """
    print(f"--- 模拟回复 (Self-Correct) ---\n{mock_response_correct}")
    parsed_grid_correct = parse_output(mock_response_correct, strategy='self_correct')
    print(f"--- 解析结果 --- \n{parsed_grid_correct}\n")

