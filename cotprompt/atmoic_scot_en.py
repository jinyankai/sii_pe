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
    list: A list of messages in OpenAI API format.
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

    # 2.3. Provide Chain-of-Thought (CoT) instructions
    user_prompt_lines.append("\n--- Instructions ---")
    user_prompt_lines.append(
        "Your task is to deduce the transformation rule from the training examples and apply it to the 'Test Input' to generate the 'Test Output'.")
    user_prompt_lines.append("\nLet's think step-by-step to find the rule:")
    user_prompt_lines.append("1. **Analyze Samples:** Look at all input-output pairs. What changes consistently?")
    user_prompt_lines.append("   - **Dimensions:** Does the output grid size differ from the input? How?")
    user_prompt_lines.append(
        "   - **Colors/Numbers:** Are colors added, removed, or changed? Which ones stay the same? Why?")
    user_prompt_lines.append(
        "   - **Objects/Sub-grids:** Can the grid be broken into objects or blocks? What happens to them?")
    user_prompt_lines.append(
        "   - **Geometric Operations:** Is the whole grid or a part of it moved (translation), rotated, or flipped (horizontal, vertical, or diagonal)?")
    user_prompt_lines.append(
        "   - **Pattern Matching:** Are specific patterns or shapes identified in the input, then modified, copied, or used to create the output?")
    user_prompt_lines.append(
        "   - **Element Analysis:** How does an element at a specific `(row, col)` in the input relate to elements in the output? Is the change based on its neighbors or its absolute position?")

    user_prompt_lines.append(
        "\n2. **Formulate Rule:** Summarize the simplest, most concise rule that explains *all* training examples.")

    user_prompt_lines.append(
        "\n3. **Verify Rule:** Critically, test your rule. Mentally apply it to *each* training input. Does it perfectly produce the corresponding training output? If not, your rule is wrong. Refine it.")

    user_prompt_lines.append(
        "\n4. **Apply Rule:** Once verified, apply this exact rule to the 'Test Input' to determine the 'Test Output'.")

    # 2.4. Specify the final output format (Structured JSON)
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
    user_prompt_lines.append("Example of the required output format (your content will be different):")
    user_prompt_lines.append("""
{
  "analysis": "Analyzed all 3 training pairs. The input grid is always 3x3. The output grid is 3x3. The rule seems to be a horizontal flip. Verified this: Example 1 input [1,2,3] becomes [3,2,1] in output, which matches. Verified for all samples.",
  "rule": "Flip the input grid horizontally to produce the output grid.",
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
    return []