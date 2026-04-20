import os
import json
import ast
import argparse
from openai import OpenAI
from tqdm import tqdm

# API 配置
BASE_URL = "https://xxxxxxxxxxxxxxxxx"
OPENAI_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
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
    parser.add_argument("--mode",      default="gpt4o", help="评估模式/输出目录")
    parser.add_argument("--input-key", default="internvl", help=f"输入文件键: {list(INPUT_PATHS.keys())}")
    return parser.parse_args()


def load_jsonl(path):
    """加载 jsonl 文件"""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_json(path):
    """加载 json 文件"""
    with open(path, 'r') as f:
        return json.load(f)


def build_eval_prompt(question, answer, pred):
    """构建评估用的 system + user prompt"""
    system_msg = (
        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
        "------"
        "##INSTRUCTIONS: "
        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
        "- Consider synonyms or paraphrases as valid matches.\n"
        "- Evaluate the correctness of the prediction compared to the answer."
    )
    
    user_msg = (
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {question}\n"
        f"Correct Answer: {answer}\n"
        f"Predicted Answer: {pred}\n\n"
        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
    )
    
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def parse_eval_response(response):
    """解析模型返回的评估结果"""
    try:
        result = ast.literal_eval(response)
        pred = result.get('pred', 'no').lower()
        score = float(result.get('score', 0))
        return pred, score
    except Exception as e:
        print(f"解析失败: {e}, response: {response}")
        return 'no', 0


def evaluate_with_api(client, question, answer, pred, model="gpt-4o-mini-2024-07-18"):
    """调用 API 进行评估"""
    messages = build_eval_prompt(question, answer, pred)
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": 32,
        "temperature": 0,  # 确定性输出
    }
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content


def main():
    args = parse_args()
    
    # 初始化 API 客户端
    client = OpenAI(base_url=BASE_URL, api_key=OPENAI_KEY)
    
    # 加载数据
    input_path = INPUT_PATHS.get(args.input_key, args.input_key)
    data_list = load_json(input_path)
    print(f"加载: {input_path}, 共 {len(data_list)} 条")
    
    # 准备输出
    os.makedirs(args.mode, exist_ok=True)
    save_file = os.path.join(args.mode, f"{args.input_key}.jsonl")
    
    # 评估循环
    results = []
    for data in tqdm(data_list, desc="Evaluating"):
        question = data['question']
        answer = data['correct_answer'][0]
        pred = data['answer']
        
        # 调用评估
        response = evaluate_with_api(client, question, answer, pred)
        pred_label, score = parse_eval_response(response) #yes/no, score0-5
        
        # 保存结果
        result = {
            "q_num": data.get("q_num"),
            "question_id": data.get("question_id"),
            "gt": answer,
            "pred": pred,
            "res": response,
            "parsed_pred": pred_label,
            "parsed_score": score,
        }
        results.append(result)
        
        # 实时打印
        print(f"\nQ: {data.get('question_id')}")
        print(f"  GT: {answer}")
        print(f"  Pred: {pred}")
        print(f"  Eval: {response}")
    
    # 保存文件
    with open(save_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f"\n结果保存: {save_file}")
    
    # 统计
    total = len(results)
    yes_count = sum(1 for r in results if r['parsed_pred'] == 'yes')
    avg_score = sum(r['parsed_score'] for r in results) / total if total else 0
    
    print(f"\n===== 统计结果 =====")
    print(f"总数: {total}")
    print(f"Yes: {yes_count}, No: {total - yes_count}")
    print(f"准确率: {yes_count / total:.4f}")
    print(f"平均分: {avg_score:.2f}")


if __name__ == "__main__":
    main()