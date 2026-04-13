import asyncio
import json
import os
import random
import re
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH       = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B"
    "/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
)
SFT_ADAPTER_PATH = "/root/autodl-tmp/final_model"
SFT_DATA_PATH    = "/root/autodl-tmp/data/finance_sft_final.json"
RESULTS_FILE     = "dpo_results.json"
EVAL_OUTPUT_JSON = "eval_results.json"
EVAL_OUTPUT_MD   = "eval_report.md"


# 确定最优 DPO adapter 路径
if not os.path.exists(RESULTS_FILE):
    print(f"错误：找不到 {RESULTS_FILE}，请先运行 train_dpo.py")
    sys.exit(1)

with open(RESULTS_FILE, encoding="utf-8") as f:
    train_results = json.load(f)

ranked = sorted(
    [(n, r) for n, r in train_results.items() if r.get("reward_accuracy") is not None],
    key=lambda x: x[1]["reward_accuracy"],
    reverse=True,
)

if not ranked:
    # fallback：按 train_loss 升序选最优
    ranked = sorted(
        [(n, r) for n, r in train_results.items() if r.get("train_loss") is not None],
        key=lambda x: x[1]["train_loss"],
    )
    print("  所有实验均无 reward_accuracy，改用 train_loss 最低作为最优配置")

if not ranked:
    print("错误：dpo_results.json 中无有效结果")
    sys.exit(1)

best_name, best_result = ranked[0]
DPO_ADAPTER_PATH = best_result["model_path"]
print(f"最优配置: {best_name}")
print(f"DPO adapter 路径: {DPO_ADAPTER_PATH}")


print("\n加载 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# 模型加载
def load_sft_model():
    print("  加载 SFT-only 模型 ...")
    base  = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH)
    model.eval()
    return model

def load_dpo_model(dpo_adapter_path: str):
    print(f"  加载 DPO 模型（adapter: {dpo_adapter_path}）...")
    base    = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    merged  = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH).merge_and_unload()
    model   = PeftModel.from_pretrained(merged, dpo_adapter_path)
    model.eval()
    return model


@torch.no_grad()
def generate(model, question: str, max_new_tokens: int = 300) -> str:
    prompt = f"### 指令:\n{question}\n\n### 回答:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out    = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# 测试集准备
print("\n准备测试集 ...")
with open(SFT_DATA_PATH, encoding="utf-8") as f:
    sft_data = json.load(f)

random.seed(2024)
test_set       = random.sample(sft_data, 50)
test_questions = [d["instruction"] for d in test_set]
print(f"测试集大小: {len(test_questions)} 条")


# 生成回答
print("\n生成 SFT-only 回答 ...")
sft_model  = load_sft_model()
sft_answers = []
for i, q in enumerate(test_questions, 1):
    ans = generate(sft_model, q)
    sft_answers.append(ans)
    if i % 10 == 0:
        print(f"  SFT 进度: {i}/50")

del sft_model
torch.cuda.empty_cache()


# 生成 DPO 回答
print("\n生成 SFT+DPO 回答 ...")
dpo_model   = load_dpo_model(DPO_ADAPTER_PATH)
dpo_answers = []
for i, q in enumerate(test_questions, 1):
    ans = generate(dpo_model, q)
    dpo_answers.append(ans)
    if i % 10 == 0:
        print(f"  DPO 进度: {i}/50")

del dpo_model
torch.cuda.empty_cache()


# 裁判
JUDGE_PROMPT = """你是一个A股投资专业评判员。请判断以下两个回答中哪一个更好。

【评判标准】
1. 是否直接回答了问题（不复述、不绕弯）
2. 是否包含具体数字或判断标准（如 "PE低于25倍" "止损5%"）
3. 是否给出可操作的建议
4. 是否避免了空洞套话（如 "需要谨慎" "综合判断"）
5. 风格是否专业、有经验感

【问题】
{question}

【回答A】
{answer_a}

【回答B】
{answer_b}

请直接输出 "A" 或 "B"，不要解释。如果两个回答质量相近，选你认为略好的那个。"""


async def judge_pair_async(client, question: str, answer_a: str, answer_b: str,
                           real_a_is_dpo: bool) -> tuple:
    for retry in range(3):
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question, answer_a=answer_a, answer_b=answer_b
                    ),
                }],
                temperature=0,
                max_tokens=10,
            )
            text = resp.choices[0].message.content.strip().upper()
            if "A" in text:
                winner = "dpo" if real_a_is_dpo else "sft"
            elif "B" in text:
                winner = "sft" if real_a_is_dpo else "dpo"
            else:
                winner = "tie"
            return winner, text
        except Exception as e:
            print(f"  裁判重试 {retry+1}/3: {e}")
            await asyncio.sleep(2)
    return "tie", "error"


async def evaluate_all() -> list:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("错误：未安装 openai 包，请 pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误：未设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    client    = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(5)  

    random.seed(2024)
    # 预先确定 A/B 顺序
    ab_flags = [random.random() < 0.5 for _ in range(50)] 

    async def evaluate_one(i: int) -> dict:
        q         = test_questions[i]
        sft_ans   = sft_answers[i]
        dpo_ans   = dpo_answers[i]
        real_a_is_dpo = ab_flags[i]

        a, b = (dpo_ans, sft_ans) if real_a_is_dpo else (sft_ans, dpo_ans)

        async with semaphore:
            winner, raw = await judge_pair_async(client, q, a, b, real_a_is_dpo)

        print(f"  [{i+1:02d}/50] winner={winner}  judge_raw={raw!r}")
        return {
            "index":      i,
            "question":   q,
            "sft_answer": sft_ans,
            "dpo_answer": dpo_ans,
            "a_was_dpo":  real_a_is_dpo,
            "winner":     winner,
            "raw_judge":  raw,
        }

    tasks   = [evaluate_one(i) for i in range(50)]
    results = await asyncio.gather(*tasks)
    return list(results)


print("\n开始 GPT-4o 裁判评估（并发 5）...")
eval_results = asyncio.run(evaluate_all())
print(f"评估完成，共 {len(eval_results)} 条")



# 计算指标
def calc_features(answers: list[str]) -> dict:
    has_num = sum(
        1 for a in answers if re.search(r'\d+[%％倍日天周月年元万亿]', a)
    )
    fillers = [
        "需要谨慎", "综合判断", "投资有风险", "密切关注",
        "结合其他指标", "不构成投资建议", "保持谨慎",
    ]
    has_filler = sum(1 for a in answers if any(f in a for f in fillers))
    avg_len    = sum(len(a) for a in answers) / len(answers)
    return {
        "avg_length":   round(avg_len, 1),
        "num_ratio":    round(has_num    / len(answers) * 100, 1),
        "filler_ratio": round(has_filler / len(answers) * 100, 1),
    }

def calc_echo_ratio(questions: list[str], answers: list[str]) -> float:
    echo_count = 0
    for q, a in zip(questions, answers):
        q_chars = set(q)
        a_start = set(a[:30])
        if q_chars and len(q_chars & a_start) / len(q_chars) > 0.5:
            echo_count += 1
    return round(echo_count / len(answers) * 100, 1)

sft_features             = calc_features(sft_answers)
sft_features["echo_ratio"] = calc_echo_ratio(test_questions, sft_answers)

dpo_features             = calc_features(dpo_answers)
dpo_features["echo_ratio"] = calc_echo_ratio(test_questions, dpo_answers)

dpo_wins = sum(1 for r in eval_results if r["winner"] == "dpo")
sft_wins = sum(1 for r in eval_results if r["winner"] == "sft")
ties     = sum(1 for r in eval_results if r["winner"] == "tie")
total    = len(eval_results)


def delta(a, b, pct=False):
    d = b - a
    return f"{d:+.1f}{'%' if pct else ''}"

report = f"""## DPO 评估结果

**最优 DPO 配置：{best_name}**

### 主指标：偏好胜率（GPT-4o 裁判，A/B 位置随机交换，n=50）

- **SFT+DPO 胜率：{dpo_wins/total*100:.1f}% ({dpo_wins}/{total})**
- SFT-only 胜率：{sft_wins/total*100:.1f}% ({sft_wins}/{total})
- 平局：{ties/total*100:.1f}% ({ties}/{total})

### 辅助指标对比

| 指标 | SFT-only | SFT+DPO | 变化 |
|------|---------|---------|------|
| 平均回答长度 | {sft_features['avg_length']} 字 | {dpo_features['avg_length']} 字 | {delta(sft_features['avg_length'], dpo_features['avg_length'])} 字 |
| 含具体数字比例 | {sft_features['num_ratio']}% | {dpo_features['num_ratio']}% | {delta(sft_features['num_ratio'], dpo_features['num_ratio'], pct=True)} |
| 含废话比例 | {sft_features['filler_ratio']}% | {dpo_features['filler_ratio']}% | {delta(sft_features['filler_ratio'], dpo_features['filler_ratio'], pct=True)} |
| 回声开头比例 | {sft_features['echo_ratio']}% | {dpo_features['echo_ratio']}% | {delta(sft_features['echo_ratio'], dpo_features['echo_ratio'], pct=True)} |

### 样本明细（前 10 条）

| # | 问题 | 胜者 | 裁判原文 |
|---|------|------|---------|
"""

for r in sorted(eval_results, key=lambda x: x["index"])[:10]:
    q_short  = r["question"][:25].replace("|", "｜") + ("…" if len(r["question"]) > 25 else "")
    report  += f"| {r['index']+1} | {q_short} | {r['winner']} | {r['raw_judge']} |\n"

report += f"""
---
*生成时间：由 eval_dpo.py 自动生成*
"""

print("\n" + report)


# 保存
with open(EVAL_OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "best_dpo_config": best_name,
            "dpo_adapter_path": DPO_ADAPTER_PATH,
            "win_rate": {
                "dpo_wins": dpo_wins,
                "sft_wins": sft_wins,
                "ties":     ties,
                "total":    total,
                "dpo_win_pct": round(dpo_wins / total * 100, 1),
                "sft_win_pct": round(sft_wins / total * 100, 1),
            },
            "features": {
                "sft": sft_features,
                "dpo": dpo_features,
            },
            "details": eval_results,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )
print(f"原始数据已保存到 {EVAL_OUTPUT_JSON}")

with open(EVAL_OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Markdown 报告已保存到 {EVAL_OUTPUT_MD}")

print("\n eval_dpo.py 全部完成")
