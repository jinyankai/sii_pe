import json
import re


def _format_grid(grid):
    """Helper function: formats a grid into a string, one list per line."""
    # 变更: 将 [1, 2, 3] 格式化为 "1 2 3"，使其更具可读性
    return "\n".join([" ".join([str(element) for element in row]) for row in grid])


def _get_grid_dims(grid):
    """Helper function: gets height and width of the grid."""
    if not grid:
        return 0, 0
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    return height, width


def construct_prompt(d):
    """
    Constructs a simplified "zero-shot" prompt for a large language model.
    Guides the model to perform CoT reasoning and output a structured JSON.
    (变更: 移除了 few-shot, 简化了 CoT 步骤)

    Args:
    d (dict): A dictionary parsed from one line of the jsonl data file.
              Note: 'd' is assumed to be pre-processed; its 'test' field
              list contains only 'input' and not the 'output' answer.

    Returns:
    list: OpenAI API的message格式列表
    """

    # 1. System Prompt: Set the model's role and task
    system_prompt = """You are an expert at solving abstract grid reasoning puzzles, specifically those from the Abstraction and Reasoning Corpus (ARC).
Your goal is to analyze training examples of grid transformations (composed of digital elements/numbers), deduce the hidden logical rule, and then apply that rule precisely to a new test input grid.
"""

    # 2. Construct the User Prompt for the *current* puzzle
    current_puzzle_lines = []
    current_puzzle_lines.append("Here is the puzzle. Please find the transformation rule.")
    current_puzzle_lines.append("\n--- Training Examples Start ---")

    # 2.1. Format all *current* training examples
    for i, example in enumerate(d['train']):
        input_grid = example['input']
        output_grid = example['output']

        in_h, in_w = _get_grid_dims(input_grid)
        out_h, out_w = _get_grid_dims(output_grid)

        current_puzzle_lines.append(f"\nExample {i + 1} Input (Height: {in_h}, Width: {in_w}):")
        current_puzzle_lines.append(_format_grid(input_grid))
        current_puzzle_lines.append(f"Example {i + 1} Output (Height: {out_h}, Width: {out_w}):")
        current_puzzle_lines.append(_format_grid(output_grid))

    current_puzzle_lines.append("\n--- Training Examples End ---")

    # 2.2. Provide the *current* test input
    current_puzzle_lines.append("\nNow, here is the test input grid you need to predict:")

    test_input_grid = d['test'][0]['input']
    test_h, test_w = _get_grid_dims(test_input_grid)

    current_puzzle_lines.append(f"\nTest Input (Height: {test_h}, Width: {test_w}):")
    current_puzzle_lines.append(_format_grid(test_input_grid))

    # 2.3. Provide *simplified* Chain-of-Thought (CoT) instructions
    current_puzzle_lines.append("\n--- Instructions ---")
    current_puzzle_lines.append(
        "Your task is to deduce the transformation rule from the training examples and apply it to the 'Test Input' to generate the 'Test Output'.")
    current_puzzle_lines.append(
        "\nLet's think step-by-step to find the rule.")

    # 简化的 CoT 引导
    current_puzzle_lines.append("\n**Step 1: Analyze Examples**")
    current_puzzle_lines.append(
        "   - Look for patterns. How do the output grids relate to the input grids?")
    current_puzzle_lines.append(
        "   - Consider dimensions, element values, shapes, positions, and relationships (e.g., 'inside', 'above', 'touching').")
    current_puzzle_lines.append(
        "   - What is the transformation type? (e.g., Element-Level, Object-Level, Decomposition-Selection, Construction).")

    # 保留自我验证
    current_puzzle_lines.append("\n**Step 2: Formulate & Verify Rule**")
    current_puzzle_lines.append(
        "   - Formulate a clear hypothesis for the rule.")
    current_puzzle_lines.append(
        "   - **CRITICAL:** Verify this rule against *every single* training example. If it fails even one, discard it and find a new one.")

    current_puzzle_lines.append("\n**Step 3: Apply Final Rule**")
    current_puzzle_lines.append(
        "   - Once you have a 100% consistent rule, apply it to the 'Test Input'.")

    # 2.4. Specify the final output format (Structured JSON)
    # (变更: 简化输出格式，仅要求网格)
    current_puzzle_lines.append("\n**Final Answer Format:**")
    current_puzzle_lines.append(
        "You must provide *only* the final predicted grid as a single, valid JSON list of lists. Do not add any other text, explanation, or JSON keys before or after the grid.")
    current_puzzle_lines.append("Example of the required output format:")
    current_puzzle_lines.append("""
[[0, 1, 0], [1, 0, 1], [0, 1, 0]]
""")

    # 3. Assemble into OpenAI API format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(current_puzzle_lines)}
    ]

    return messages


def parse_output(text):
    """
    Parses the LLM's output text to extract the predicted grid.
    (变更: 此版本仅查找原始的 JSON 列表 [[...]])

    Args:
    text (str): The full output text from the large language model.

    Returns:
    list: The predicted grid (a 2D list).
          Returns an empty list [] if parsing fails.
    """
    try:
        # Find the last occurrence of a JSON list of lists.
        # re.DOTALL (re.S) allows '.' to match newlines
        matches = re.findall(r"(\[\[.*?\]\])", text, re.DOTALL)
        if matches:
            # Assume the last match is the intended answer
            json_string = matches[-1]

            # Clean up potential markdown code fences
            json_string = json_string.strip().lstrip('```json').lstrip('```').rstrip('```')

            parsed_list = json.loads(json_string)

            # Basic validation of the extracted grid
            if isinstance(parsed_list, list):
                if not parsed_list or all(isinstance(row, list) for row in parsed_list):
                    return parsed_list
    except (json.JSONDecodeError, re.error, Exception) as e:
        # print(f"Error parsing output: {e}") # Can be uncommented for debugging
        pass  # Parsing failure

    return []

