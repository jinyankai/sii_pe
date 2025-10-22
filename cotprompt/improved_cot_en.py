import json
import re


def construct_prompt(d):
    """
    Constructs a prompt for a large language model (Chain-of-Thought version - English)

    Args:
    d (dict): A dictionary parsed from one line of the jsonl data file.
              Note: 'd' is assumed to be pre-processed; its 'test' field
              list contains only 'input' and not the 'output' answer.

    Returns:
    list: A list of messages in OpenAI API format, allowing for multi-turn prompts.
    """

    def format_grid(grid):
        """Helper function: formats a grid into a string, one list per line."""
        return "\n".join([str(row) for row in grid])

    # 1. System Prompt: Set the model's role and task
    system_prompt = """You are an expert at solving abstract visual reasoning puzzles, specifically from the ARC (Abstraction and Reasoning Corpus).
Your goal is to analyze several training examples of grid transformations, deduce the hidden logical rule, and then apply that rule precisely to a new test input grid.
"""

    # 2. User Prompt: Build CoT instructions
    user_prompt_lines = []
    user_prompt_lines.append("Here are several training examples. Please find the transformation rule.")
    user_prompt_lines.append("\n--- Training Examples Start ---")

    # 2.1. Format all training examples
    for i, example in enumerate(d['train']):
        input_grid = format_grid(example['input'])
        output_grid = format_grid(example['output'])
        user_prompt_lines.append(f"\nExample {i + 1} Input:")
        user_prompt_lines.append(input_grid)
        user_prompt_lines.append(f"\nExample {i + 1} Output:")
        user_prompt_lines.append(output_grid)

    user_prompt_lines.append("\n--- Training Examples End ---")

    # 2.2. Provide the test input
    user_prompt_lines.append("\nNow, here is the test input grid you need to predict:")
    user_prompt_lines.append(f"\nTest Input:")
    test_input_grid = format_grid(d['test'][0]['input'])
    user_prompt_lines.append(test_input_grid)

    # 2.3. Provide Chain-of-Thought (CoT) instructions
    # (Change: CoT steps are optimized and ordered)
    user_prompt_lines.append("\n--- Instructions ---")
    user_prompt_lines.append("Your task is to deduce the transformation rule from the training examples and apply it to the 'Test Input' to generate the 'Test Output'.")
    user_prompt_lines.append("\nLet's think step-by-step to find the rule:")
    user_prompt_lines.append("1. **Analyze Examples:** Look at all input-output pairs. What is the consistent change? (Please try in order)")
    user_prompt_lines.append(
        "   - **A. Element Analysis (Local):** What is the relationship between an element at a specific position `(row, col)` in the input and the output? Is this change based on its neighbors or its absolute position?")
    user_prompt_lines.append(
        "   - **B. Color/Pattern Matching (Object):** Are any colors (numbers) added, removed, or changed? Are specific patterns or shapes identified in the input, then modified, copied, or used to create the output?")
    user_prompt_lines.append(
        "   - **C. Geometry/Dimensions (Global):** Is the output grid size different? Is the entire grid, or a part of it, moved (translated), rotated, or flipped (horizontally, vertically, or diagonally)?")
    user_prompt_lines.append(
        "   - **D. Construction/Algorithm (Complex):** (If the above rules do not apply) Is the output *constructed from scratch*? Is there an algorithmic rule (e.g., 'tile the input pattern A N times, where N is the input height')?")


    user_prompt_lines.append("\n2. **Formulate Rule:** Summarize the most concise rule that explains *all* training examples.")

    user_prompt_lines.append("\n3. **Apply Rule:** Apply this rule to the 'Test Input' to produce the 'Test Output'.")

    # 2.4. Specify the final output format
    user_prompt_lines.append("\n**Final Answer Format:**")
    user_prompt_lines.append(
        "After your reasoning analysis, please provide *only* the final predicted output grid, formatted as a 2D list (JSON array). Do not add any other text or explanation after the final grid.")
    user_prompt_lines.append("Example of the final answer format: [[0, 1, 0], [1, 0, 1], [0, 1, 0]]")

    user_content = "\n".join(user_prompt_lines)

    # 3. Assemble into OpenAI API format
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def parse_output(text):
    """
    Parses the large language model's output text to extract the predicted grid.

    Args:
    text (str): The output text from the large language model under the designed prompt.

    Returns:
    list: A 2D array (Python list) parsed from the output text.
          Returns an empty list [] if parsing fails.
    """
    try:
        # Use regex to find all 2D list-like strings
        # re.DOTALL (re.S) allows '.' to match newlines, handling JSON arrays that span multiple lines
        # '.*?' is a non-greedy match
        matches = re.findall(r"(\[\[.*?\]\])", text, re.DOTALL)

        if matches:
            # Assume the model might output the correct answer last, so we take the last match
            json_string = matches[-1]

            # Attempt to parse this string as a Python list
            parsed_list = json.loads(json_string)

            # Basic type validation
            if isinstance(parsed_list, list):
                # Ensure it's a list of lists (or an empty list)
                if not parsed_list or all(isinstance(row, list) for row in parsed_list):
                    return parsed_list

    except (json.JSONDecodeError, re.error, Exception):
        # Catch JSON parsing errors, regex errors, or any other exceptions
        # print(f"Error parsing output: {e}") # Can be uncommented for debugging
        pass

    # If no match is found, or if parsing fails, return an empty list
    return []

