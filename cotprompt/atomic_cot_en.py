import json
import re


def construct_prompt(d):
    """
    Constructs a prompt for a large language model (Structured CoT version).
    Guides the model to perform CoT reasoning and output a structured JSON.

    Args:
    d (dict): A dictionary parsed from one line of the jsonl data file.
              Note: 'd' is assumed to be pre-processed; its 'test' field
              list contains only 'input' and not the 'output' answer.

    Returns:
    list: OpenAI API的message格式列表
    """

    def format_grid(grid):
        """Helper function: formats a grid into a string, one list per line."""
        return "\n".join([str(row) for row in grid])

    def get_grid_dims(grid):
        """Helper function: gets height and width of the grid."""
        if not grid:
            return 0, 0
        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        return height, width

    # 1. System Prompt: Set the model's role and task
    system_prompt = """You are an expert at solving abstract visual reasoning puzzles, specifically those from the Abstraction and Reasoning Corpus (ARC).
Your goal is to analyze training examples of grid transformations, deduce the hidden logical rule, and then apply that rule precisely to a new test input grid.
"""

    # 2. User Prompt: Construct the CoT guidance
    user_prompt_lines = []
    user_prompt_lines.append("Here are several training examples. Please find the transformation rule.")
    user_prompt_lines.append("\n--- Training Examples Start ---")

    # 2.1. Format all training examples
    for i, example in enumerate(d['train']):
        input_grid = example['input']
        output_grid = example['output']

        in_h, in_w = get_grid_dims(input_grid)
        out_h, out_w = get_grid_dims(output_grid)

        user_prompt_lines.append(f"\nExample {i + 1} Input (Height: {in_h}, Width: {in_w}):")
        user_prompt_lines.append(format_grid(input_grid))
        user_prompt_lines.append(f"Example {i + 1} Output (Height: {out_h}, Width: {out_w}):")
        user_prompt_lines.append(format_grid(output_grid))

    user_prompt_lines.append("\n--- Training Examples End ---")

    # 2.2. Provide the test input
    user_prompt_lines.append("\nNow, here is the test input grid you need to predict:")

    test_input_grid = d['test'][0]['input']
    test_h, test_w = get_grid_dims(test_input_grid)

    user_prompt_lines.append(f"\nTest Input (Height: {test_h}, Width: {test_w}):")
    user_prompt_lines.append(format_grid(test_input_grid))

    # 2.3. [!! 已修改 !!] 提供更广泛的分析策略 (v3, 术语修正)
    user_prompt_lines.append("\n--- Instructions ---")
    user_prompt_lines.append(
        "Your task is to deduce the transformation rule from the training examples and apply it to the 'Test Input' to generate the 'Test Output'.")
    user_prompt_lines.append(
        "\nLet's think step-by-step to find the rule. Start with high-level analysis and drill down.")

    user_prompt_lines.append("\n**Step 1: High-Level Analysis (Holistic View)**")
    user_prompt_lines.append(
        "   - **Transformation Type:** Is this a element-level modification, an object-based transformation, or a decomposition-selection task?")
    user_prompt_lines.append(
        "   - **Decomposition-Selection Check:** Does the output look like an *exact copy* of a *part* (sub-grid, object, or block) from the input? (e.g., input is 9x3, output is 3x3). If YES, the task is to find the *selection rule* (e.g., 'select the block with the most 5s', 'select the block with the lowest sum').")
    user_prompt_lines.append(
        "   - **Geometric Operations:** Is there a global transformation? (e.g., rotate 90 degrees, flip vertically, mirror).")
    user_prompt_lines.append(
        "   - **Symmetry & Repetition:** Does the output complete a pattern, remove asymmetry, or tile a shape from the input?")

    user_prompt_lines.append("\n**Step 2: Object-Level Analysis (If not Decomposition-Selection)**")
    user_prompt_lines.append(
        "   - **Object Identification:** Can the grid be broken into distinct objects (shapes made of the same number/value)?")
    user_prompt_lines.append(
        "   - **Object Fate:** What happens to each object? (e.g., moved, number changed, scaled, distorted, deleted, copied).")
    user_prompt_lines.append(
        "   - **Object Relationships:** Does the transformation depend on relationships *between* objects? (e.g., 'move the small shape inside the large one', 'draw a line between the two 8s').")

    user_prompt_lines.append("\n**Step 3: Element-Level Analysis (If no clear object rule)**")
    user_prompt_lines.append(
        "   - **Dimensions:** How do output dimensions `(H_out, W_out)` relate to input `(H_in, W_in)`? (e.g., same, `2*H_in`, `H_in / 2`).")
    user_prompt_lines.append(
        "   - **Element Mapping:** How does an output element at `(r, c)` get its value? Is it from a corresponding input element `(r, c)` or `(r/2, c/2)`?")
    user_prompt_lines.append(
        "   - **Neighborhood Logic:** Does the value of an output element `(r, c)` depend on the *neighbors* of the input element `(r, c)`? (e.g., 'cell becomes 1 if it has > 3 neighbors with value 2').")
    user_prompt_lines.append(
        "   - **Number/Value Logic:** Are numbers/values being systematically changed? (e.g., 'all 2s become 4s', 'the most frequent number becomes 8').")

    user_prompt_lines.append("\n**Step 4: Formulate & Verify Rule**")
    user_prompt_lines.append(
        "   - **Formulate Hypothesis:** Based on Steps 1-3, state a simple, clear hypothesis for the rule.")
    user_prompt_lines.append(
        "   - **Verify Hypothesis:** **CRITICAL STEP.** Apply your rule hypothesis to *every single* training example. Does it perfectly predict the output for *all* of them? If it fails on even *one* example, your hypothesis is wrong. Discard it and find a new one.")

    user_prompt_lines.append("\n**Step 5: Apply Final Rule**")
    user_prompt_lines.append(
        "   - Once you have a rule that is 100% consistent across all training examples, apply that exact rule to the 'Test Input' to generate the 'Test Output'.")

    # 2.4. Specify the final output format (Structured JSON)
    user_prompt_lines.append("\n**Final Answer Format:**")
    user_prompt_lines.append(
        "You must provide *only* a single, valid JSON object as your final answer. Do not add any text or explanation before or after the JSON block.")
    user_prompt_lines.append("The JSON object must have the following structure:")
    user_prompt_lines.append("""
{
  "predicted_grid": [[...]]
}
""")
    user_prompt_lines.append("Example of the required output format (your content will be different):")
    user_prompt_lines.append("""
{
  "predicted_grid": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
}
""")

    user_content = "\n".join(user_prompt_lines)

    # 3. Assemble into OpenAI API format
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


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
