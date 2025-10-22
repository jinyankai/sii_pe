import json
import os
import re
import asyncio
# 移除 aiohttp，导入 openai 相关的库
from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIConnectionError
from collections import Counter

# 导入考生需要实现的两个函数
# (我假设 construct_cot.py 与 run.py 在同一目录下)
# 注意：根据你的目录结构，这个导入可能需要调整
# 假设 construct_cot.py 在当前目录
from cotprompt.atomic_cot import construct_prompt, parse_output

# from cotprompt.atomic_cot  import construct_prompt, parse_output # 原始导入

# --- 配置区 ---
API_KEY = "sk-5d1d1180313c45589472a340afe4a5f5"  # 请替换为你的 API Key
MODEL_NAME = "deepseek-chat"
# 注意：使用 openai 库时，需要提供 base_url
API_BASE_URL = "https://api.deepseek.com"  # deepseek 的基础 URL
VALIDATION_FILE = "val.jsonl"

# --- 快速测试配置 ---
# ... (保持不变) ...
QUICK_TEST = False  # 设置为 True 以快速测试单个任务
QUICK_TEST_TASK_INDEX = 0  # 测试 val.jsonl 中的第几个任务 (从 0 开始)

# --- 异步配置 ---
# ... (保持不变) ...
NUM_SAMPLES = 10  # 每个任务的采样次数 (将并行执行)
TEMPERATURE = 1.0
MAX_CONCURRENT_REQUESTS = 20  # 同时允许的最大 API 请求数
RETRY_ATTEMPTS = 3  # API 请求失败时的重试次数
RETRY_DELAY = 2  # 每次重试的延迟时间 (秒)
API_TIMEOUT = 60  # API 请求超时时间 (秒)
MAX_TOKENS = 8000  # 最大 token 保持 8000


# --- 核心函数 ---

def load_data(file_path, quick_test=False, task_index=0):
    """从jsonl文件加载数据 (保持不变)"""
    # ... (函数体保持不变) ...
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


async def async_call_llm_api(client, semaphore, messages, temperature, model_name):
    """(异步) 调用大语言模型API (使用 AsyncOpenAI)，带有限流和重试"""

    # payload 现在在 create 方法中定义
    # headers 由 openai 客户端自动处理

    # 在信号量控制下执行
    async with semaphore:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                # 使用 AsyncOpenAI 客户端发起请求
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=MAX_TOKENS,
                    timeout=API_TIMEOUT
                )
                # print("ans from llm :", response.choices[0].message.content) # 调试时使用
                return response.choices[0].message.content

            # 捕获 openai 特定的异常
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                print(f"API 调用失败 (尝试 {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return None  # 所有重试均失败
            except Exception as e:
                # 捕获其他意外错误
                print(f"API 调用中发生意外错误 (尝试 {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return None


def evaluate_prediction(prediction, ground_truth):
    """评估预测是否与真实答案完全匹配 (保持不变)"""
    # ... (函数体保持不变) ...
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


async def run_evaluation(tasks, api_key, base_url, num_samples, temperature):
    """(异步) 运行完整评估流程 (使用 AsyncOpenAI)"""
    total_correct = 0
    total_tasks = len(tasks)

    # 创建信号量以限制并发数
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # 创建 AsyncOpenAI 客户端实例
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # aiohttp.ClientSession 已被移除
    # async with aiohttp.ClientSession() as session: # (已移除)

    # 注意：AsyncOpenAI 客户端实例通常不需要像 session 一样使用 'async with'
    # 它内部管理连接池，但我们需要确保在完成后正确关闭
    # 不过对于这个脚本的生命周期，在 main 结束时自动关闭即可
    # 或者我们可以在这里使用 async with，如果 openai 库支持的话
    # 查阅文档后，推荐的方式是在 main 函数管理 client 生命周期，或者在这里管理

    # 为了保持结构类似，我们假设 client 在这个函数内有效
    # (或者在 main 中创建并传入)
    # 这里我们在函数内部创建它

    try:
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
            api_tasks = []
            for j in range(num_samples):
                api_tasks.append(
                    asyncio.create_task(
                        # 传递 client 和 semaphore
                        async_call_llm_api(client, semaphore, messages, temperature, MODEL_NAME)
                    )
                )

            print(f"  - 正在为 {num_samples} 个样本并行发起 API 请求...")
            raw_outputs = await asyncio.gather(*api_tasks)
            # --- 并行结束 ---

            predictions = []
            for j, raw_output in enumerate(raw_outputs):
                if raw_output:
                    if QUICK_TEST:
                        print(f"  - [快速测试] Sample {j + 1} 原始输出:\n{raw_output}")

                    parsed_grid = parse_output(raw_output)
                    # print(parsed_grid)

                    if parsed_grid and isinstance(parsed_grid, list):
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
    finally:
        # 确保在使用完后关闭客户端（尽管在这个脚本中可能不是严格必须）
        # await client.close() # AsyncOpenAI 似乎没有公开的 close()
        pass

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
    if not API_BASE_URL:
        print("Error: API_BASE_URL 未在脚本中设置 (使用 AsyncOpenAI 时需要)。")
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
        # 将 API_KEY 和 API_BASE_URL 传递给 run_evaluation
        await run_evaluation(validation_tasks, API_KEY, API_BASE_URL, num_samples, temperature)


# --- 主程序入口 ---
if __name__ == "__main__":
    # 使用 asyncio.run() 来启动异步主函数
    asyncio.run(main())
