import json
import re


def _format_grid(grid):
    """Helper function: formats a grid into a string, one list per line."""
    return "\n".join([str(row) for row in grid])


def _get_grid_dims(grid):
    """Helper function: gets height and width of the grid."""
    if not grid:
        return 0, 0
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    return height, width


def _get_few_shot_example_messages():
    """
    [FEW-SHOT DEMONSTRATION]
    Returns a hardcoded, solved ARC puzzle example (662c240a).
    This example demonstrates the full "Tree of Thoughts" (ToT) reasoning
    process and the expected structured JSON output.

    (变更: 这里的 CoT 步骤已按“局部->整体”的顺序重新排列)
    """

    # --- Few-Shot Example: User's Request ---
    # (This is the puzzle data you provided: 662c240a)
    user_request = """Here are several training examples. Please find the transformation rule.

--- Training Examples Start ---

Example 1 Input (Height: 9, Width: 3):
[[2, 2, 2], [2, 2, 3], [2, 3, 3], [5, 7, 7], [7, 5, 5], [7, 5, 5], [8, 8, 1], [1, 8, 1], [1, 8, 1]]
Example 1 Output (Height: 3, Width: 3):
[[8, 8, 1], [1, 8, 1], [1, 8, 1]]

Example 2 Input (Height: 9, Width: 3):
[[1, 5, 5], [5, 1, 1], [5, 1, 1], [3, 3, 3], [3, 6, 3], [3, 6, 6], [7, 7, 7], [7, 2, 2], [7, 2, 2]]
Example 2 Output (Height: 3, Width: 3):
[[3, 3, 3], [3, 6, 3], [3, 6, 6]]

Example 3 Input (Height: 9, Width: 3):
[[8, 8, 4], [4, 4, 4], [4, 4, 8], [1, 1, 3], [1, 3, 3], [3, 3, 1], [6, 2, 2], [2, 2, 2], [2, 2, 6]]
Example 3 Output (Height: 3, Width: 3):
[[8, 8, 4], [4, 4, 4], [4, 4, 8]]

Example 4 Input (Height: 9, Width: 3):
[[8, 9, 8], [9, 8, 8], [8, 8, 8], [2, 2, 1], [2, 2, 1], [1, 1, 2], [4, 4, 4], [4, 4, 3], [3, 3, 3]]
Example 4 Output (Height: 3, Width: 3):
[[4, 4, 4], [4, 4, 3], [3, 3, 3]]

--- Training Examples End ---

Now, here is the test input grid you need to predict:

Test Input (Height: 9, Width: 3):
[[5, 4, 4], [4, 5, 4], [4, 5, 4], [3, 3, 2], [3, 3, 2], [2, 2, 3], [1, 1, 1], [1, 8, 8], [1, 8, 8]]

--- Instructions ---
Your task is to deduce the transformation rule from the training examples and apply it to the 'Test Input' to generate the 'Test Output'.

Let's think step-by-step to find the rule. Start with local analysis and build up.

**Step 1: Element-Level Analysis (Local View)**
   - **Dimensions:** How do output dimensions `(H_out, W_out)` relate to input `(H_in, W_in)`? (e.g., same, `2*H_in`, `H_in / 2`).
   - **Element Mapping:** How does an output element at `(r, c)` get its value? Is it from a corresponding input element `(r, c)` or `(r/2, c/2)`?
   - **Neighborhood Logic:** Does the value of an output element `(r, c)` depend on the *neighbors* of the input element `(r, c)`? (e.g., 'element becomes 1 if it has > 3 neighbors with value 2'). (变更: 'cell' 替换为 'element')
   - **Number/Value Logic:** Are numbers/values being systematically changed? (e.g., 'all 2s become 4s', 'the most frequent number becomes 8').

**Step 2: Object-Level Analysis (Grouping Elements)**
   - (If no clear element-level rule is found)
   - **Object Identification:** Can the grid be broken into distinct objects (shapes made of the same number/value)?
   - **Object Fate:** What happens to each object? (e.g., moved, number changed, scaled, distorted, deleted, copied).
   - **Object Relationships:** Does the transformation depend on relationships *between* objects? (e.g., 'move the small shape inside the large one', 'draw a line between the two 8s').

**Step 3: High-Level Analysis (Holistic View)**
   - (If no clear object-level rule is found)
   - **Transformation Type:** Based on steps 1 & 2, is this a element-level modification, an object-based transformation, or a decomposition-selection task?
   - **Decomposition-Selection Check:** Does the output look like an *exact copy* of a *part* (sub-grid, object, or block) from the input? (e.g., input is 9x3, output is 3x3). If YES, the task is to find the *selection rule* (e.g., 'select the block with the most 5s', 'select the block with the lowest sum').
   - **Geometric Operations:** Is there a global transformation? (e.g., rotate 90 degrees, flip vertically, mirror).
   - **Symmetry & Repetition:** Does the output complete a pattern, remove asymmetry, or tile a shape from the input?

**Step 4: Formulate & Verify Rule**
   - **Formulate Hypothesis:** Based on Steps 1-3, state a simple, clear hypothesis for the rule.
   - **Verify Hypothesis:** **CRITICAL STEP.** Apply your rule hypothesis to *every single* training example. Does it perfectly predict the output for *all* of them? If it fails on even *one* example, your hypothesis is wrong. Discard it and find a new one.

**Step 5: Apply Final Rule**
   - Once you have a rule that is 100% consistent across all training examples, apply that exact rule to the 'Test Input' to generate the 'Test Output'.

**Final Answer Format:**
You must provide *only* a single, valid JSON object as your final answer. Do not add any text or explanation before or after the JSON block.
The JSON object must have the following structure:
{
  "analysis": "A brief, step-by-step analysis following the steps above. Explicitly state which hypotheses were tested and why they were kept or discarded. If it's a selection task, list the properties of each sub-grid.",
  "rule": "A concise, one-sentence statement of the final, verified transformation rule.",
  "predicted_grid": [[...]]
}
Example of the required output format (your content will be different):
{
  "predicted_grid": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
}
"""

    # --- Few-Shot Example: Assistant's Response ---
    # (变更: 这里的分析步骤已更新, 以匹配新的 "局部->整体" CoT)
    assistant_response = """
{
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
    Constructs a prompt for a large language model (Structured CoT version).
    Guides the model to perform CoT reasoning and output a structured JSON.
    (变更: 这里的 System Prompt 和 CoT 步骤已更新)

    Args:
    d (dict): A dictionary parsed from one line of the jsonl data file.
              Note: 'd' is assumed to be pre-processed; its 'test' field
              list contains only 'input' and not the 'output' answer.

    Returns:
    list: OpenAI API的message格式列表
    """

    # 1. System Prompt: Set the model's role and task
    # (变更: 从 "visual" 变为 "grid", 增加 "digital elements")
    system_prompt = """You are an expert at solving abstract grid reasoning puzzles, specifically those from the Abstraction and Reasoning Corpus (ARC).
Your goal is to analyze training examples of grid transformations (composed of digital elements/numbers), deduce the hidden logical rule, and then apply that rule precisely to a new test input grid.
"""

    # 2. Get Few-Shot Example Messages
    # This includes one full 'user' turn and one full 'assistant' turn
    # demonstrating the desired "Local-to-Global" ToT reasoning.
    messages = _get_few_shot_example_messages()

    # 3. Construct the User Prompt for the *current* puzzle
    current_puzzle_lines = []
    current_puzzle_lines.append("\n\n--- New Puzzle ---")
    current_puzzle_lines.append("Excellent. Here is the next puzzle. Please use the exact same analysis method.")
    current_puzzle_lines.append("\n--- Training Examples Start ---")

    # 3.1. Format all *current* training examples
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

    # 3.2. Provide the *current* test input
    current_puzzle_lines.append("\nNow, here is the test input grid you need to predict:")

    test_input_grid = d['test'][0]['input']
    test_h, test_w = _get_grid_dims(test_input_grid)

    current_puzzle_lines.append(f"\nTest Input (Height: {test_h}, Width: {test_w}):")
    current_puzzle_lines.append(_format_grid(test_input_grid))

    # 3.3. Provide Chain-of-Thought (CoT) instructions (identical to the few-shot example)
    # (变更: CoT 步骤已按“局部->整体”的顺序重新排列)
    current_puzzle_lines.append("\n--- Instructions ---")
    current_puzzle_lines.append(
        "Your task is to deduce the transformation rule from the training examples and apply it to the 'Test Input' to generate the 'Test Output'.")
    current_puzzle_lines.append(
        "\nLet's think step-by-step to find the rule. Start with local analysis and build up.")  # (变更: 措辞调整)

    current_puzzle_lines.append("\n**Step 1: Element-Level Analysis (Local View)**")
    current_puzzle_lines.append(
        "   - **Dimensions:** How do output dimensions `(H_out, W_out)` relate to input `(H_in, W_in)`? (e.g., same, `2*H_in`, `H_in / 2`).")
    current_puzzle_lines.append(
        "   - **Element Mapping:** How does an output element at `(r, c)` get its value? Is it from a corresponding input element `(r, c)` or `(r/2, c/2)`?")
    current_puzzle_lines.append(
        "   - **Neighborhood Logic:** Does the value of an output element `(r, c)` depend on the *neighbors* of the input element `(r, c)`? (e.g., 'element becomes 1 if it has > 3 neighbors with value 2').")  # (变更: 'cell' 替换为 'element')
    current_puzzle_lines.append(
        "   - **Number/Value Logic:** Are numbers/values being systematically changed? (e.g., 'all 2s become 4s', 'the most frequent number becomes 8').")

    current_puzzle_lines.append("\n**Step 2: Object-Level Analysis (Grouping Elements)**")
    current_puzzle_lines.append("   - (If no clear element-level rule is found)")
    current_puzzle_lines.append(
        "   - **Object Identification:** Can the grid be broken into distinct objects (shapes made of the same number/value)?")
    current_puzzle_lines.append(
        "   - **Object Fate:** What happens to each object? (e.g., moved, number changed, scaled, distorted, deleted, copied).")
    current_puzzle_lines.append(
        "   - **Object Relationships:** Does the transformation depend on relationships *between* objects? (e.g., 'move the small shape inside the large one', 'draw a line between the two 8s').")

    current_puzzle_lines.append("\n**Step 3: High-Level Analysis (Holistic View)**")
    current_puzzle_lines.append("   - (If no clear object-level rule is found)")
    current_puzzle_lines.append(
        "   - **Transformation Type:** Based on steps 1 & 2, is this a element-level modification, an object-based transformation, or a decomposition-selection task?")
    current_puzzle_lines.append(
        "   - **Decomposition-Selection Check:** Does the output look like an *exact copy* of a *part* (sub-grid, object, or block) from the input? (e.g., input is 9x3, output is 3x3). If YES, the task is to find the *selection rule* (e.g., 'select the block with the most 5s', 'select the block with the lowest sum').")
    current_puzzle_lines.append(
        "   - **Geometric Operations:** Is there a global transformation? (e.g., rotate 90 degrees, flip vertically, mirror).")
    current_puzzle_lines.append(
        "   - **Symmetry & Repetition:** Does the output complete a pattern, remove asymmetry, or tile a shape from the input?")

    current_puzzle_lines.append("\n**Step 4: Formulate & Verify Rule**")
    current_puzzle_lines.append(
        "   - **Formulate Hypothesis:** Based on Steps 1-3, state a simple, clear hypothesis for the rule.")
    current_puzzle_lines.append(
        "   - **Verify Hypothesis:** **CRITICAL STEP.** Apply your rule hypothesis to *every single* training example. Does it perfectly predict the output for *all* of them? If it fails on even *one* example, your hypothesis is wrong. Discard it and find a new one.")

    current_puzzle_lines.append("\n**Step 5: Apply Final Rule**")
    current_puzzle_lines.append(
        "   - Once you have a rule that is 100% consistent across all training examples, apply that exact rule to the 'Test Input' to generate the 'Test Output'.")

    # 3.4. Specify the final output format (Structured JSON)
    current_puzzle_lines.append("\n**Final Answer Format:**")
    current_puzzle_lines.append(
        "You must provide *only* a single, valid JSON object as your final answer. Do not add any text or explanation before or after the JSON block.")
    current_puzzle_lines.append("The JSON object must have the following structure:")
    current_puzzle_lines.append("""
{
  "predicted_grid": [[...]]
}
""")
    current_puzzle_lines.append("Example of the required output format (your content will be different):")
    current_puzzle_lines.append("""
{
  "predicted_grid": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
}
""")

    # 4. Append the current puzzle prompt as the final 'user' message
    messages.append({
        "role": "user",
        "content": "\n".join(current_puzzle_lines)
    })

    # 5. Assemble into OpenAI API format (add system prompt at the beginning)
    return [
        {"role": "system", "content": system_prompt},
    ] + messages


def parse_output(text):
    """
    Parses the LLM's output text to extract the predicted grid from a structured JSON.
    This version expects a JSON object like:
    {
      "analysis": "...",
      "rule": "...",
      "predicted_grid": [[...]]
    }

    Args:
    text (str): The full output text from the large language model.

    Returns:
    list: The "predicted_grid" (a 2D list) extracted from the JSON.
          Returns an empty list [] if parsing fails or the key is not found.
    """
    try:
        # Use regex to find all JSON objects (non-greedy match)
        # re.DOTALL (re.S) allows '.' to match newlines
        matches = re.findall(r"(\{.*?\})", text, re.DOTALL)

        if matches:
            # Assume the last JSON object is the intended answer
            json_string = matches[-1]

            # Clean up potential markdown code fences
            json_string = json_string.strip().lstrip('```json').lstrip('```').rstrip('```')

            # Attempt to parse the string as JSON
            parsed_json = json.loads(json_string)

            # Check if it's a dictionary and has the required key
            if isinstance(parsed_json, dict):
                predicted_grid = parsed_json.get("predicted_grid")

                # Basic validation of the extracted grid
                if isinstance(predicted_grid, list):
                    # Ensure it's a list of lists (or an empty list)
                    if not predicted_grid or all(isinstance(row, list) for row in predicted_grid):
                        return predicted_grid

    except (json.JSONDecodeError, re.error, Exception) as e:
        # Catch JSON parsing errors, regex errors, or any other exceptions
        # print(f"Error parsing output: {e}") # Can be uncommented for debugging
        pass

    # If no match, parsing fails, or key/format is incorrect
    # 尝试回退到原始 `atomic_cot.py` 的解析逻辑
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
