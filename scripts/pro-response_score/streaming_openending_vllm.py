#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 参考文档
# https://github.com/datawhalechina/self-llm/blob/master/models/Qwen2.5/03-Qwen2.5-7B-Instruct%20vLLM%20%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md
# https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html

# ==================== 多进程启动方式配置（必须放在最开头） ====================
# 解决多卡推理时 CUDA fork 问题：Cannot re-initialize CUDA in forked subprocess
import multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

import os
import json
import ast
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


RESULT_DIR = "~"

# 输入文件路径配置（注释保留，方便切换）
INPUT_PATHS = {
    "internvl":   f"{RESULT_DIR}/internvl.json",
    "flash":      f"{RESULT_DIR}/flash.json",
    "vc2":        f"{RESULT_DIR}/vc2.json",
    "llavavideo": f"{RESULT_DIR}/llavavideo.json",
}

def parse_args():
    parser = argparse.ArgumentParser(description="视频问答评估")
    parser.add_argument("--mode",      default="online", help="评估模式/输出目录")
    parser.add_argument("--input-key", default="internvl", help=f"输入文件键: {list(INPUT_PATHS.keys())}")
    return parser.parse_args()


def load_jsonl(file_path):
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def create_evaluation_messages(question, answer, pred, strict_mode=False, loose_mode=False):
    """
    创建评估用的消息列表
    
    Args:
        question: 问题
        answer: 正确答案
        pred: 预测答案
        strict_mode: 是否使用严格评估模式
        loose_mode: 是否使用宽松评估模式
    """
    if strict_mode:
        # 严格评估模式：要求预测答案必须包含正确答案的所有关键事实
        system_content = (
            "You are an evaluator for video description tasks. "
            "Your job is to judge whether the predicted answer fully includes the key factual content of the correct answer. "
            "------\n"
            "## STRICT EVALUATION RULES:\n"
            "- The predicted answer must contain ALL essential elements from the correct answer (e.g., subject, action, object, location, time if specified).\n"
            "- Paraphrasing is acceptable ONLY if all key facts are preserved.\n"
            "- Adding extra details is allowed and does not penalize the score.\n"
            "- Missing, generalizing, or replacing any key element (e.g., 'dog' → 'animal') results in a low score.\n"
            "- Score 5: perfect inclusion of all key facts (exact or paraphrased with full fidelity).\n"
            "- Score ≤2: missing or distorting core information."
        )
    elif loose_mode:
        # 宽松评估模式：允许部分匹配
        system_content = (
            "You are an evaluator for video description tasks. "
            "Your job is to judge whether the predicted answer fully includes the key factual content of the correct answer. "
            "------\n"
            "## STRICT EVALUATION RULES:\n"
            "- Focus on whether the predicted answer captures the main idea or at least one key event from the correct answer.\n"
            "- Partial matches are acceptable.\n"
            "- Generalizations (e.g., 'dog' → 'animal') may be tolerated if context is preserved.\n"
            "- Score 5: near-perfect match; Score 3: partial but meaningful overlap; Score 1–2: vague or irrelevant."
        )
    else:
        # 默认模式：关注语义匹配
        system_content = (
            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
            "------"
            "##INSTRUCTIONS: "
            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
            "- Consider synonyms or paraphrases as valid matches.\n"
            "- Evaluate the correctness of the prediction compared to the answer."
        )
    
    messages = [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": (
                "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n"
                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
            )
        }
    ]
    return messages


def evaluate_with_vllm(model, tokenizer, messages, sampling_params):
    """
    使用 vLLM 进行评估
    
    Args:
        model: vLLM 模型实例
        tokenizer: 分词器
        messages: 消息列表
        sampling_params: 采样参数
    
    Returns:
        模型生成的响应文本
    """
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 生成输出
    outputs = model.generate([text], sampling_params=sampling_params)
    response = outputs[0].outputs[0].text
    
    return response


def compute_metrics(save_file):
    """
    计算评估指标
    
    Args:
        save_file: 保存结果的文件路径
    
    Returns:
        包含各项指标的字典
    """
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    
    combined_contents = load_jsonl(save_file)
    
    for item in combined_contents:
        count += 1
        # 解析分数
        score_match = ast.literal_eval(item["res"])['score']
        score = int(score_match)
        score_sum += score
        
        # 统计 yes/no
        pred = ast.literal_eval(item["res"])['pred']
        if "yes" in pred.lower():
            yes_count += 1
        elif "no" in pred.lower():
            no_count += 1
    
    average_score = score_sum / count if count > 0 else 0
    accuracy = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    
    metrics = {
        "yes_count": yes_count,
        "no_count": no_count,
        "accuracy": accuracy,
        "average_score": average_score,
        "total_samples": count
    }
    
    return metrics


def main():
    # 解析命令行参数
    args = parse_args()
    
    # ==================== 模型路径配置 ====================
    # 可选的模型路径（保留所有原始注释路径）
    
    # 72B 模型路径
    # model_path = "Qwen2.5-72B-Instruct"
    model_path = "Qwen2.5-7B-Instruct"
    
    # ==================== vLLM 模型初始化 ====================
    model = LLM(
        model=model_path, 
        tensor_parallel_size=1, #8
        max_model_len=1024, 
        enforce_eager=True, 
        disable_custom_all_reduce=True
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # ==================== 采样参数配置 ====================
    # 默认采样参数
    sampling_params = SamplingParams(
        temperature=0, 
        top_p=0.01, 
        max_tokens=64, 
        stop_token_ids=[1699, 151329, 151336, 151338]
    )
    
    # ==================== 数据文件配置 ====================
    # 可选的预测结果文件路径（保留所有原始注释路径）
    
    # 当前使用的文件
    # pred_json = "/mnt/agents/output/eval_vllm.py"
    
    # 其他可选文件（注释掉）
    pred_json = INPUT_PATHS[arg.input_key]
    
    # 加载预测数据
    pred_contents = load_json(pred_json)
    print(f"加载文件: {pred_json}, 样本数: {len(pred_contents)}")
    
    # ==================== 输出目录配置 ====================
    save_dir = args.mode
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = os.path.join(save_dir, pred_json.split("/").pop() + "l")
    
    print(f"结果将保存至: {save_file}")
    
    # ==================== 评估流程 ====================
    count = 0
    
    with open(save_file, "w") as f:
        for data in tqdm(pred_contents[:200], desc="Evaluating"):
            question = data['question']
            answer = data['correct_answer'][0]
            pred = data["answer"]
            
            # 创建评估消息（默认模式）
            # 可选：strict_mode=True 启用严格模式，loose_mode=True 启用宽松模式
            messages = create_evaluation_messages(
                question=question,
                answer=answer,
                pred=pred,
                strict_mode=False,
                loose_mode=False
            )
            
            # 使用 vLLM 进行评估
            try:
                response = evaluate_with_vllm(
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    sampling_params=sampling_params
                )
            except Exception as e:
                print(f"评估出错 (question_id: {data.get('question_id', 'unknown')}): {e}")
                response = "{'pred': 'no', 'score': 0}"
            
            # 构建输出数据
            new_data = {
                "q_num": data["q_num"],
                "gt": data['correct_answer'][0],
                "pred": pred,
                "res": response
            }
            
            # 打印调试信息
            print(f"\nQuestion ID: {data.get('question_id', 'N/A')}")
            print(f"GT: {new_data['gt']}")
            print(f"Pred: {new_data['pred']}")
            print(f"Response: {response}")
            print("-" * 50, flush=True)
            
            # 保存结果
            json.dump(new_data, f)
            f.write('\n')
            f.flush()
            count += 1
    
    print(f"\n评估完成，共处理 {count} 个样本")
    
    # ==================== 计算并输出指标 ====================
    metrics = compute_metrics(save_file)
    
    print("\n" + "=" * 50)
    print("评估结果:")
    print(f"Yes count: {metrics['yes_count']}")
    print(f"No count: {metrics['no_count']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Average score: {metrics['average_score']:.4f}")
    print(f"Total samples: {metrics['total_samples']}")
    print("=" * 50)


if __name__ == "__main__":
    main()