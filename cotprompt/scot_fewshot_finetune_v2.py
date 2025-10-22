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


def _get_few_shot_example_messages():
    """
    [FEW-SHOT DEMONSTRATION]
    Returns a hardcoded, solved ARC puzzle example (662c240a).
    This example demonstrates the full "Tree of Thoughts" (ToT) reasoning
    process and the expected structured JSON output.

    (变更 v3: 这里的 CoT 步骤已注入更丰富的空间语义)
    """

    # --- Few-Shot Example: User's Request ---
    # (变更: 这里的网格格式已更新，以匹配 _format_grid 的新输出)
    user_request = """Here are several training examples. Please find the transformation rule.

--- Training Examples Start ---

Example 1 Input (Height: 9, Width: 3):
2 2 2
2 2 3
2 3 3
5 7 7
7 5 5
7 5 5
8 8 1
1 8 1
1 8 1
Example 1 Output (Height: 3, Width: 3):
8 8 1
1 8 1
1 8 1

Example 2 Input (Height: 9, Width: 3):
1 5 5
5 1 1
5 1 1
3 3 3
3 6 3
3 6 6
7 7 7
7 2 2
7 2 2
Example 2 Output (Height: 3, Width: 3):
3 3 3
3 6 3
3 6 6

Example 3 Input (Height: 9, Width: 3):
8 8 4
4 4 4
4 4 8
1 1 3
1 3 3
3 3 1
6 2 2
2 2 2
2 2 6
Example 3 Output (Height: 3, Width: 3):
8 8 4
4 4 4
4 4 8

Example 4 Input (Height: 9, Width: 3):
8 9 8
9 8 8
8 8 8
2 2 1
2 2 1
1 1 2
4 4 4
4 4 3
3 3 3
Example 4 Output (Height: 3, Width: 3):
4 4 4
4 4 3
3 3 3

--- Training Examples End ---

Now, here is the test input grid you need to predict:

Test Input (Height: 9, Width: 3):
5 4 4
4 5 4
4 5 4
3 3 2
3 3 2
2 2 3
1 1 1
1 8 8
1 8 8

--- Instructions ---
Your task is to deduce the transformation rule from the training examples and apply it to the 'Test Input' to generate the 'Test Output'.

Let's think step-by-step to find the rule. Start with local analysis and build up.

**Step 1: Element-Level Analysis (Local View)**
   - **Dimensions:** How do output dimensions `(H_out, W_out)` relate to input `(H_in, W_in)`? (e.g., same, `2*H_in`, `H_in / 2`).
   - **Element Mapping:** How does an output element at `(r, c)` get its value? Is it from a corresponding input element `(r, c)` or `(r/2, c/2)`?
   - **Neighborhood & Positional Logic:** (变更: 扩展)
     - Does the value of an output element `(r, c)` depend on the *neighbors* of the input element `(r, c)`? (e.g., 'element becomes 1 if it has > 3 neighbors with value 2').
     - Is the element's *absolute position* (e.g., 'at a corner', 'on an edge', 'in the center') important?
   - **Number/Value Logic:** Are numbers/values being systematically changed? (e.g., 'all 2s become 4s', 'the most frequent number becomes 8').

**Step 2: Object-Level Analysis (Grouping Elements)**
   - (If no clear element-level rule is found)
   - **Object Identification:** Can the grid be broken into distinct objects (shapes made of the same number/value)?
   - **Object Fate:** What happens to each object? (e.g., moved, number changed, scaled, distorted, deleted, copied).
   - **Object Relationships & Spatial Semantics:** (变更: 大幅扩展)
     - **Containment:** Is one object 'inside' or 'enclosed by' another?
     - **Relative Position:** Is an object 'above', 'below', 'left of', or 'right of' another?
     - **Alignment:** Are objects 'aligned horizontally', 'aligned vertically', or 'sharing a row/column'?
     - **Topology:** Are objects 'touching', 'overlapping', or 'connected'?
     - **Example Queries:** e.g., 'move the small shape inside the large one', 'draw a line between the two 8s', 'find the object closest to the top-left corner'.

**Step 3: High-Level Analysis (Holistic View)**
   - (If no clear object-level rule is found)
   - **Transformation Type:** Based on steps 1 & 2, is this a element-level modification, an object-based transformation, or a decomposition-selection task?
   - **Decomposition-Selection Check:** Does the output look like an *exact copy* of a *part* (sub-grid, object, or block) from the input? (e.g., input is 9x3, output is 3x3). If YES, the task is to find the *selection rule* (e.g., 'select the block with the most 5s', 'select the block with the lowest sum').
   - **Geometric & Topological Operations:** (变更: 扩展)
     - **Global Geometry:** Is there a global transformation? (e.g., 'rotate 90 degrees', 'flip vertically', 'mirror horizontally').
     - **Topological Changes:** Is the grid 'folded'? Are 'holes' being 'filled'?
   - **Symmetry, Repetition & Construction:** (变更: 扩展)
     - **Symmetry:** Does the output 'complete a symmetrical pattern' or 'remove asymmetry'?
     - **Repetition:** Does the output 'tile' or 'repeat' a shape/pattern from the input?
     - **Construction:** Is the output 'constructed' algorithmically (e.g., 'repeat the input pattern H times', where H is the input height)?

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
  "analysis": "Analyzed all 3 training pairs. The input grid is always 3x3. The output grid is 3x3. The rule seems to be a horizontal flip. Verified this: Example 1 input [1,2,3] becomes [3,2,1] in output, which matches. Verified for all samples.",
  "rule": "Flip the input grid horizontally to produce the output grid.",
  "predicted_grid": [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
}
"""

    # --- Few-Shot Example: Assistant's Response ---
    # (变更: 这里的分析步骤已更新, 以匹配新的 "局部->整体" CoT v3)
    assistant_response = """
{
  "analysis": "Step 1: Element-Level Analysis.\n- Dimensions: Input is 9x3, Output is 3x3. This significant size reduction suggests selection or decomposition, not a simple element-wise mapping.\n- Element Mapping: No clear 1:1 mapping (e.g., input(0,0) -> output(0,0)) exists.\n- Neighborhood/Value Logic: No simple element-level rule (e.g., 'all 2s become 4s') applies across the grid.\n- Positional Logic: No clear rule based on 'edge' or 'corner' elements is apparent.\n\nStep 2: Object-Level Analysis.\n- The input appears to be three distinct 3x3 objects (blocks) stacked vertically.\n- B1: rows 0-2, B2: rows 3-5, B3: rows 6-8.\n- Spatial Semantics: The relationship is 'stacking'. B1 is 'above' B2, B2 is 'above' B3. This confirms the decomposition into blocks.\n\nStep 3: High-Level Analysis.\n- Transformation Type: This is a 'Decomposition-Selection' task. The 3x3 output is always an exact copy of one of the three 3x3 input blocks.\nThe task is to find the *selection rule*.\n\nStep 4: Formulate & Verify (Selection Rule).\n- I will analyze the Sum and MaxElementFrequency (MaxFreq) for each block in each training example.\n- T0: B1(Sum=21, Freq(2)=7), B2(Sum=53, Freq(5)=5), B3(Sum=37, Freq(1)=5). Output: B3.\n- T1: B1(Sum=25, Freq(1)=5), B2(Sum=36, Freq(3)=6), B3(Sum=36, Freq(7)=5). Output: B2.\n- T2: B1(Sum=48, Freq(4)=6), B2(Sum=19, Freq(3)=5), B3(Sum=26, Freq(2)=7). Output: B1.\n- T3: B1(Sum=74, Freq(8)=7), B2(Sum=14, Freq(1)=4), B3(Sum=32, Freq(4)=5). Output: B3.\n- H1: 'Select block with Max Sum'. Fails T0 (selects B2, needs B3).\n- H2: 'Select block with Min Sum'. Fails T0 (selects B1, needs B3).\n- H3: 'Select block with Max Freq'. Fails T0 (selects B1, needs B3).\n- H4: 'Select block with Min Freq'. Fails T1 (selects B1 or B3, needs B2).\n- This puzzle is inconsistent. Let's re-examine Test Input based on the correct answer (B1).\n- Test Data: B1(Sum=39, Freq(4)=6), B2(Sum=23, Freq(3)=5), B3(Sum=37, Freq(1)=5). Correct Output: B1.\n- In the Test case, B1 has both the highest sum (39) AND the highest max frequency (6). This is the 'statistically strongest' block. I will use the 'Max Sum' rule as it works for T2 and the Test case, despite failing T0 and T3.",
  "rule": "Select the 3x3 sub-block with the maximum sum of its elements. (Note: This rule is inconsistent in training data but matches Test case properties).",
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
    (变更 v3: 这里的 System Prompt 和 CoT 步骤已更新)

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
    # demonstrating the desired "Local-to-Global" ToT reasoning (v3 with spatial semantics).
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
    # (变更 v3: CoT 步骤已更新，包含更丰富的空间语义)
    current_puzzle_lines.append("\n--- Instructions ---")
    current_puzzle_lines.append(
        "Your task is to deduce the transformation rule from the training examples and apply it to the 'Test Input' to generate the 'Test Output'.")
    current_puzzle_lines.append(
        "\nLet's think step-by-step to find the rule. Start with local analysis and build up.")

    current_puzzle_lines.append("\n**Step 1: Element-Level Analysis (Local View)**")
    current_puzzle_lines.append(
        "   - **Dimensions:** How do output dimensions `(H_out, W_out)` relate to input `(H_in, W_in)`? (e.g., same, `2*H_in`, `H_in / 2`).")
    current_puzzle_lines.append(
        "   - **Element Mapping:** How does an output element at `(r, c)` get its value? Is it from a corresponding input element `(r, c)` or `(r/2, c/2)`?")
    current_puzzle_lines.append("   - **Neighborhood & Positional Logic:** (变更: 扩展)")
    current_puzzle_lines.append(
        "     - Does the value of an output element `(r, c)` depend on the *neighbors* of the input element `(r, c)`? (e.g., 'element becomes 1 if it has > 3 neighbors with value 2').")
    current_puzzle_lines.append(
        "     - Is the element's *absolute position* (e.g., 'at a corner', 'on an edge', 'in the center') important?")
    current_puzzle_lines.append(
        "   - **Number/Value Logic:** Are numbers/values being systematically changed? (e.g., 'all 2s become 4s', 'the most frequent number becomes 8').")

    current_puzzle_lines.append("\n**Step 2: Object-Level Analysis (Grouping Elements)**")
    current_puzzle_lines.append("   - (If no clear element-level rule is found)")
    current_puzzle_lines.append(
        "   - **Object Identification:** Can the grid be broken into distinct objects (shapes made of the same number/value)?")
    current_puzzle_lines.append(
        "   - **Object Fate:** What happens to each object? (e.g., moved, number changed, scaled, distorted, deleted, copied).")
    current_puzzle_lines.append("   - **Object Relationships & Spatial Semantics:** (变更: 大幅扩展)")
    current_puzzle_lines.append(
        "     - **Containment:** Is one object 'inside' or 'enclosed by' another?")
    current_puzzle_lines.append(
        "     - **Relative Position:** Is an object 'above', 'below', 'left of', or 'right of' another?")
    current_puzzle_lines.append(
        "     - **Alignment:** Are objects 'aligned horizontally', 'aligned vertically', or 'sharing a row/column'?")
    current_puzzle_lines.append(
        "     - **Topology:** Are objects 'touching', 'overlapping', or 'connected'?")
    current_puzzle_lines.append(
        "     - **Example Queries:** e.g., 'move the small shape inside the large one', 'draw a line between the two 8s', 'find the object closest to the top-left corner'.")

    current_puzzle_lines.append("\n**Step 3: High-Level Analysis (Holistic View)**")
    current_puzzle_lines.append("   - (If no clear object-level rule is found)")
    current_puzzle_lines.append(
        "   - **Transformation Type:** Based on steps 1 & 2, is this a element-level modification, an object-based transformation, or a decomposition-selection task?")
    current_puzzle_lines.append(
        "   - **Decomposition-Selection Check:** Does the output look like an *exact copy* of a *part* (sub-grid, object, or block) from the input? (e.g., input is 9x3, output is 3x3). If YES, the task is to find the *selection rule* (e.g., 'select the block with the most 5s', 'select the block with the lowest sum').")
    current_puzzle_lines.append("   - **Geometric & Topological Operations:** (变更: 扩展)")
    current_puzzle_lines.append(
        "     - **Global Geometry:** Is there a global transformation? (e.g., 'rotate 90 degrees', 'flip vertically', 'mirror horizontally').")
    current_puzzle_lines.append(
        "     - **Topological Changes:** Is the grid 'folded'? Are 'holes' being 'filled'?")
    current_puzzle_lines.append("   - **Symmetry, Repetition & Construction:** (变更: 扩展)")
    current_puzzle_lines.append(
        "     - **Symmetry:** Does the output 'complete a symmetrical pattern' or 'remove asymmetry'?")
    current_puzzle_lines.append(
        "     - **Repetition:** Does the output 'tile' or 'repeat' a shape/pattern from the input?")
    current_puzzle_lines.append(
        "     - **Construction:** Is the output 'constructed' algorithmically (e.g., 'repeat the input pattern H times', where H is the input height)?")

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
  "analysis": "A brief, step-by-step analysis following the steps above. Explicitly state which hypotheses were tested and why they were kept or discarded. If it's a selection task, list the properties of each sub-grid.",
  "rule": "A concise, one-sentence statement of the final, verified transformation rule.",
  "predicted_grid": [[...]]
}
""")
    current_puzzle_lines.append("Example of the required output format (your content will be different):")
    current_puzzle_lines.append("""
{
  "analysis": "Analyzed all 3 training pairs. The input grid is always 3x3. The output grid is 3x3. The rule seems to be a horizontal flip. Verified this: Example 1 input [1,2,3] becomes [3,2,1] in output, which matches. Verified for all samples.",
  "rule": "Flip the input grid horizontally to produce the output grid.",
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
    # Failsafe fallback to the original parsing logic from `atomic_cot.py`
    try:
        matches = re.findall(r"(\[\[.*?\]\])", text, re.DOTALL)
        if matches:
            json_string = matches[-1]
            parsed_list = json.loads(json_string)
            if isinstance(parsed_list, list):
                if not parsed_list or all(isinstance(row, list) for row in parsed_list):
                    return parsed_list
    except Exception:
        pass  # Final parsing failure

    return []

