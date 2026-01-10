import datasets

dataset = datasets.load_dataset("openai/gsm8k", "main")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

import re

def extract_solution(solution_str):
    """提取数值答案"""
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    return solution.group(0).split("#### ")[1].replace(",", "")

def process_fn(example, idx, split):
    question_raw = example["question"]
    answer_raw = example["answer"]
    
    # 添加推理指令
    question = question_raw + ' Let\'s think step by step and output the final answer after "####".'
    solution = extract_solution(answer_raw)
    
    return {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": question}], # 对话prompt
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution}, 
        # 这里指出reward使用的一个字段"ground_truth"，会被传入reward函数
        "extra_info": {"split": split, "index": idx, "answer": answer_raw, "question": question_raw}
    }

# 转换数据
train_dataset = train_dataset.map(lambda x, idx: process_fn(x, idx, "train"), with_indices=True)
test_dataset = test_dataset.map(lambda x, idx: process_fn(x, idx, "test"), with_indices=True)

import os
train_dataset.to_parquet(os.path.join("/vepfs_c/zhaowenxuan/workspace/vldoc_agent_rl/code/verl/data", "train.parquet"))
test_dataset.to_parquet(os.path.join("/vepfs_c/zhaowenxuan/workspace/vldoc_agent_rl/code/verl/data", "test.parquet"))