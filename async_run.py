import json
import os
import re
import asyncio  # 导入 asyncio
import aiohttp  # 导入 aiohttp
from collections import Counter

# 导入考生需要实现的两个函数
# (我假设 construct_cot.py 与 run.py 在同一目录下)
from cotprompt.construct_cot import construct_prompt, parse_output

# --- 配置区 ---
API_KEY = "sk-5d1d1180313c45589472a340afe4a5f5"  # 请替换为你的 API Key
MODEL_NAME = "deepseek-chat"
API_URL = "https://api.deepseek.com/chat/completions"
VALIDATION_FILE = "val.jsonl"

# --- 快速测试配置 ---
QUICK_TEST = False  # 设置为 True 以快速测试单个任务
QUICK_TEST_TASK_INDEX = 0  # 测试 val.jsonl 中的第几个任务 (从 0 开始)

# --- 异步配置 ---
# (快速测试模式会覆盖以下设置)
NUM_SAMPLES = 5  # 每个任务的采样次数 (将并行执行)
TEMPERATURE = 1.0
MAX_CONCURRENT_REQUESTS = 10  # 同时允许的最大 API 请求数
RETRY_ATTEMPTS = 3  # API 请求失败时的重试次数
RETRY_DELAY = 2  # 每次重试的延迟时间 (秒)
API_TIMEOUT = 60  # API 请求超时时间 (秒)
MAX_TOKENS = 8000


# --- 核心函数 ---

def load_data(file_path, quick_test=False, task_index=0):
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

        return tasks
    except FileNotFoundError:
        print(f"Error: 验证文件 '{file_path}' 未找到。")
        return []


async def async_call_llm_api(session, semaphore, messages, api_key, temperature, model_name):
    """(异步) 调用大语言模型API，带有限流和重试"""
    if not api_key:
        raise ValueError("API key not found.")

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

    # 在信号量控制下执行
    async with semaphore:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                # 使用 aiohttp session 发起 POST 请求
                async with session.post(API_URL, json=payload, headers=headers, timeout=API_TIMEOUT) as response:
                    response.raise_for_status()  # 如果状态码是 4xx 或 5xx, 抛出异常
                    result = await response.json()
                    # print("ans from llm :",result['choices'][0]['message']['content']) # 调试时使用
                    return result['choices'][0]['message']['content']
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"API 调用失败 (尝试 {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return None  # 所有重试均失败
            except Exception as e:
                print(f"API 调用中发生意外错误: {e}")
                return None


def evaluate_prediction(prediction, ground_truth):
    """评估预测是否与真实答案完全匹配 (保持不变)"""
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


async def run_evaluation(tasks, api_key, num_samples, temperature):
    """(异步) 运行完整评估流程"""
    total_correct = 0
    total_tasks = len(tasks)

    # 创建信号量以限制并发数
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    # 创建一个 aiohttp 客户端会话 (用于连接池)
    async with aiohttp.ClientSession() as session:

        for i, task in enumerate(tasks):
            print(f"--- Processing Task {i + 1}/{total_tasks} ---")

            # 构造不含答案的测试样本
            task_for_prompt = task.copy()
            task_for_prompt['test'] = [{'input': task['test'][0]['input']}]

            # 构造提示词 (同步操作)
            messages = construct_prompt(task_for_prompt)
            if QUICK_TEST:
                print("--- [快速测试] Prompt 内容: ---")
                print(json.dumps(messages, indent=2, ensure_ascii=False))
                print("---------------------------------")

            # --- 并行执行 ---
            # 为 N 次采样创建所有异步任务
            api_tasks = []
            for j in range(num_samples):
                api_tasks.append(
                    asyncio.create_task(
                        async_call_llm_api(session, semaphore, messages, api_key, temperature, MODEL_NAME)
                    )
                )

            print(f"  - 正在为 {num_samples} 个样本并行发起 API 请求...")
            # 等待所有任务完成
            raw_outputs = await asyncio.gather(*api_tasks)
            # --- 并行结束 ---

            predictions = []
            for j, raw_output in enumerate(raw_outputs):
                if raw_output:
                    if QUICK_TEST:
                        print(f"  - [快速测试] Sample {j + 1} 原始输出:\n{raw_output}")

                    parsed_grid = parse_output(raw_output)

                    if parsed_grid and isinstance(parsed_grid, list):
                        # 将列表转换为元组，以便Counter可以哈希
                        predictions.append(tuple(map(tuple, parsed_grid)))
                    else:
                        print(f"  - Sample {j + 1} 解析失败。")
                else:
                    print(f"  - Sample {j + 1} API 请求失败。")

            if not predictions:
                print("  - Result: INCORRECT (没有有效的解析结果)")
                print(f"    Ground Truth: {task['test'][0]['output']}")
                continue

            # 自洽性投票 (逻辑不变)
            vote_counts = Counter(predictions)
            most_common_result = vote_counts.most_common(1)

            most_common_prediction_as_tuple = most_common_result[0][0]
            final_prediction = list(map(list, most_common_prediction_as_tuple))

            ground_truth = task['test'][0]['output']
            score = evaluate_prediction(final_prediction, ground_truth)

            if score == 1:
                total_correct += 1
                print(f"  - Result: CORRECT (投票数: {most_common_result[0][1]}/{num_samples})")
            else:
                print("  - Result: INCORRECT")
                print(f"    Predicted: {final_prediction}")
                print(f"    Ground Truth: {ground_truth}")

    accuracy = total_correct / total_tasks if total_tasks > 0 else 0
    print(f"\n--- Evaluation Finished ---")
    print(f"Total Tasks: {total_tasks}")
    print(f"Correct Predictions: {total_correct}")
    print(f"Final Accuracy (with Self-Consistency): {accuracy:.2%}")


async def main():
    """异步主函数"""
    if not API_KEY:
        print("Error: API_KEY 未在脚本中设置。")
        return

    # 根据是否快速测试来调整参数
    if QUICK_TEST:
        num_samples = 1
        temperature = 0.0  # 快速测试时使用 0 温度确保结果一致
        print("[快速测试模式] 已激活: 1个任务, 1次采样, 温度 0.0")
    else:
        num_samples = NUM_SAMPLES
        temperature = TEMPERATURE
        print(f"[完整评估模式] 已激活: {NUM_SAMPLES}次采样, 温度 {TEMPERATURE}")

    validation_tasks = load_data(
        VALIDATION_FILE,
        quick_test=QUICK_TEST,
        task_index=QUICK_TEST_TASK_INDEX
    )

    if validation_tasks:
        await run_evaluation(validation_tasks, API_KEY, num_samples, temperature)


# --- 主程序入口 ---
if __name__ == "__main__":
    # 使用 asyncio.run() 来启动异步主函数
    asyncio.run(main())

