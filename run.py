import json
import os
import re
import time
import urllib.request
import urllib.error
from collections import Counter

# 导入考生需要实现的两个函数
from cotprompt.construct_cot import construct_prompt, parse_output

# --- 配置区 ---
API_KEY =  "sk-5d1d1180313c45589472a340afe4a5f5"
MODEL_NAME = "deepseek-chat"
API_URL = "https://api.deepseek.com/chat/completions"
VALIDATION_FILE = "val.jsonl"
NUM_SAMPLES = 1  # 每个任务的采样次数，用于自洽性投票
TEMPERATURE = 1.0
MAX_TOKENS = 8000  # 根据任务复杂度调整


# --- 核心函数 ---

def load_data(file_path, quick_test=False, task_index=19):
    """从jsonl文件加载数据 (增强版)"""
    tasks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    # 尝试解析每一行
                    tasks.append(json.loads(line))
                except json.JSONDecodeError:
                    # 如果某一行JSON格式错误，则跳过并打印提示
                    print(f"警告: 跳过 {file_path} 的第 {i + 1} 行，因其JSON格式无效。")

        if not tasks:
            print(f"Error: 文件 '{file_path}' 为空或所有行均无效。")
            return []

        if quick_test:
            if 0 <= task_index < len(tasks):
                print(f"[快速测试模式] 已启动，仅加载任务 {task_index} (从有效行中选取)")
                return [tasks[task_index]]
            else:
                print(f"Error: 快速测试索引 {task_index} 超出范围 (共 {len(tasks)} 个有效任务)")
                return []
        # print(tasks)
        return tasks
    except FileNotFoundError:
        print(f"Error: 验证文件 '{file_path}' 未找到。")
        return []


def call_llm_api(messages, api_key, temperature, model_name):
    """调用大语言模型API"""
    if not api_key:
        raise ValueError("API key not found. Please set the DEEPSEEK_API_KEY environment variable.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(API_URL, data=data, headers=headers)

    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            # print("ans from llm :",result['choices'][0]['message']['content'])
            return result['choices'][0]['message']['content']
    except urllib.error.HTTPError as e:
        print(f"Error calling API: {e.code} {e.reason}")
        print(e.read().decode())
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def evaluate_prediction(prediction, ground_truth):
    """评估预测是否与真实答案完全匹配"""
    if not isinstance(prediction, list) or not isinstance(ground_truth, list):
        return 0
    if len(prediction) != len(ground_truth):
        return 0
    for i in range(len(ground_truth)):
        if not isinstance(prediction[i], list) or len(prediction[i]) != len(ground_truth[i]):
            return 0
        if prediction[i] != ground_truth[i]:
            return 0
    return 1


def run_evaluation(tasks, api_key, num_samples):
    """运行完整评估流程"""
    total_correct = 0
    total_tasks = len(tasks)

    for i, task in enumerate(tasks):
        print(f"--- Processing Task {i + 1}/{total_tasks} ---")

        # 构造不含答案的测试样本
        task_for_prompt = task.copy()
        task_for_prompt['test'] = [{'input': task['test'][0]['input']}]

        # 构造提示词
        messages = construct_prompt(task_for_prompt)

        predictions = []
        for j in range(num_samples):
            print(f"  - Sample {j + 1}/{num_samples}")
            raw_output = call_llm_api(messages, api_key, TEMPERATURE, MODEL_NAME)
            if raw_output:
                parsed_grid = parse_output(raw_output)

                print("LLM Output:",parsed_grid)
                if parsed_grid and isinstance(parsed_grid, list):
                    # 将列表转换为元组，以便Counter可以哈希
                    predictions.append(tuple(map(tuple, parsed_grid)))
            time.sleep(1)  # 避免API速率限制

        if not predictions:
            print("  - No valid predictions generated.")
            continue

            # ... ( vote_counts = Counter(predictions) 之后 ) ...

        # 自洽性投票
        vote_counts = Counter(predictions)
        most_common_result = vote_counts.most_common(1)  # 这会返回一个列表，例如：[ (prediction_tuple, count) ]

        if not most_common_result:
            print("  - 投票失败，没有有效预测。")
            continue  # 跳过这个任务

        # [!! 关键修复 !!]
        # 1. 从列表中取出第一个元素 (prediction_tuple, count)
        # 2. 再从该元组中取出第 0 个元素，即 prediction_tuple 本身
        most_common_prediction_as_tuple = most_common_result[0][0]

        # 3. 现在才将这个正确的元组 (它是一个元组的元组) 转换回列表的列表
        final_prediction = list(map(list, most_common_prediction_as_tuple))

        ground_truth = task['test'][0]['output']
        score = evaluate_prediction(final_prediction, ground_truth)
        # ... (后续代码不变) ...

        if score == 1:
            total_correct += 1
            print("  - Result: CORRECT")
        else:
            print("  - Result: INCORRECT")
            print(f"    Predicted: {final_prediction}")
            print(f"    Ground Truth: {ground_truth}")

    accuracy = total_correct / total_tasks if total_tasks > 0 else 0
    print(f"\n--- Evaluation Finished ---")
    print(f"Total Tasks: {total_tasks}")
    print(f"Correct Predictions: {total_correct}")
    print(f"Final Accuracy (with Self-Consistency): {accuracy:.2%}")


# --- 主程序入口 ---
if __name__ == "__main__":
    if not API_KEY:
        print("Error: DEEPSEEK_API_KEY environment variable not set.")
    else:
        validation_tasks = load_data(VALIDATION_FILE)
        run_evaluation(validation_tasks, API_KEY, NUM_SAMPLES)