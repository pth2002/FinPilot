# -*- coding: utf-8 -*-

import asyncio
import json
import os
import random
import re
import sys


_DIR = os.path.dirname(os.path.abspath(__file__))
ANSWERS_PATH = os.path.join(_DIR, "eval_answers.json")
OUTPUT_JSON  = os.path.join(_DIR, "eval_results.json")
OUTPUT_MD    = os.path.join(_DIR, "eval_report.md")

if not os.path.exists(ANSWERS_PATH):
    print(f"错误：找不到 {ANSWERS_PATH}，请先在 AutoDL 上运行 generate_answers.py 并下载结果")
    sys.exit(1)

with open(ANSWERS_PATH, encoding="utf-8") as f:
    answers = json.load(f)

print(f"读取回答文件：{len(answers)} 条")
test_questions = [r["question"]   for r in answers]
sft_answers    = [r["sft_answer"] for r in answers]
dpo_answers    = [r["dpo_answer"] for r in answers]


# OpenAI 客户端
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("错误：未设置 OPENAI_API_KEY 环境变量")
    sys.exit(1)

try:
    from openai import AsyncOpenAI
except ImportError:
    print("错误：未安装 openai 包，请 pip install openai")
    sys.exit(1)

client = AsyncOpenAI(api_key=api_key)


# 4o 裁判
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


async def judge_pair(question: str, answer_a: str, answer_b: str,
                     real_a_is_dpo: bool) -> tuple:
    """返回 (winner: 'dpo'/'sft'/'tie', raw_response: str)"""
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
    semaphore = asyncio.Semaphore(5)

    random.seed(2024)
    ab_flags = [random.random() < 0.5 for _ in range(len(answers))]

    async def evaluate_one(i: int) -> dict:
        q             = test_questions[i]
        sft_ans       = sft_answers[i]
        dpo_ans       = dpo_answers[i]
        real_a_is_dpo = ab_flags[i]
        a, b          = (dpo_ans, sft_ans) if real_a_is_dpo else (sft_ans, dpo_ans)

        async with semaphore:
            winner, raw = await judge_pair(q, a, b, real_a_is_dpo)

        print(f"  [{i+1:02d}/{len(answers)}] winner={winner}  raw={raw!r}")
        return {
            "index":      i,
            "question":   q,
            "sft_answer": sft_ans,
            "dpo_answer": dpo_ans,
            "a_was_dpo":  real_a_is_dpo,
            "winner":     winner,
            "raw_judge":  raw,
        }

    tasks   = [evaluate_one(i) for i in range(len(answers))]
    results = await asyncio.gather(*tasks)
    return list(results)


print("\n开始 GPT-4o 裁判评估（并发 5）...")
eval_results = asyncio.run(evaluate_all())
print(f"评估完成，共 {len(eval_results)} 条")


# 指标
def calc_features(answers: list) -> dict:
    has_num    = sum(1 for a in answers if re.search(r'\d+[%％倍日天周月年元万亿]', a))
    fillers    = ["需要谨慎", "综合判断", "投资有风险", "密切关注",
                  "结合其他指标", "不构成投资建议", "保持谨慎"]
    has_filler = sum(1 for a in answers if any(f in a for f in fillers))
    avg_len    = sum(len(a) for a in answers) / len(answers)
    return {
        "avg_length":   round(avg_len, 1),
        "num_ratio":    round(has_num    / len(answers) * 100, 1),
        "filler_ratio": round(has_filler / len(answers) * 100, 1),
    }

def calc_echo_ratio(questions: list, answers: list) -> float:
    echo_count = 0
    for q, a in zip(questions, answers):
        q_chars = set(q)
        a_start = set(a[:30])
        if q_chars and len(q_chars & a_start) / len(q_chars) > 0.5:
            echo_count += 1
    return round(echo_count / len(answers) * 100, 1)

sft_features               = calc_features(sft_answers)
sft_features["echo_ratio"] = calc_echo_ratio(test_questions, sft_answers)
dpo_features               = calc_features(dpo_answers)
dpo_features["echo_ratio"] = calc_echo_ratio(test_questions, dpo_answers)


# 统计
dpo_wins = sum(1 for r in eval_results if r["winner"] == "dpo")
sft_wins = sum(1 for r in eval_results if r["winner"] == "sft")
ties     = sum(1 for r in eval_results if r["winner"] == "tie")
total    = len(eval_results)


# 报告
def delta(a, b):
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}"

report = f"""## DPO 评估结果

### 主指标：偏好胜率（GPT-4o 裁判，A/B 位置随机交换，n={total}）

- **SFT+DPO 胜率：{dpo_wins/total*100:.1f}% ({dpo_wins}/{total})**
- SFT-only 胜率：{sft_wins/total*100:.1f}% ({sft_wins}/{total})
- 平局：{ties/total*100:.1f}% ({ties}/{total})

### 辅助指标对比

| 指标 | SFT-only | SFT+DPO | 变化 |
|------|---------|---------|------|
| 平均回答长度 | {sft_features['avg_length']} 字 | {dpo_features['avg_length']} 字 | {delta(sft_features['avg_length'], dpo_features['avg_length'])} 字 |
| 含具体数字比例 | {sft_features['num_ratio']}% | {dpo_features['num_ratio']}% | {delta(sft_features['num_ratio'], dpo_features['num_ratio'])}% |
| 含废话比例 | {sft_features['filler_ratio']}% | {dpo_features['filler_ratio']}% | {delta(sft_features['filler_ratio'], dpo_features['filler_ratio'])}% |
| 回声开头比例 | {sft_features['echo_ratio']}% | {dpo_features['echo_ratio']}% | {delta(sft_features['echo_ratio'], dpo_features['echo_ratio'])}% |

### 样本明细（前 10 条）

| # | 问题 | 胜者 | 裁判原文 |
|---|------|------|---------|
"""

for r in sorted(eval_results, key=lambda x: x["index"])[:10]:
    q_short = r["question"][:25].replace("|", "｜")
    if len(r["question"]) > 25:
        q_short += "…"
    report += f"| {r['index']+1} | {q_short} | {r['winner']} | {r['raw_judge']} |\n"

report += "\n---\n*由 local_judge.py 生成*\n"

print("\n" + report)


# 保存
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "win_rate": {
                "dpo_wins":    dpo_wins,
                "sft_wins":    sft_wins,
                "ties":        ties,
                "total":       total,
                "dpo_win_pct": round(dpo_wins / total * 100, 1),
                "sft_win_pct": round(sft_wins / total * 100, 1),
            },
            "features": {"sft": sft_features, "dpo": dpo_features},
            "details":  eval_results,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )
print(f"原始数据已保存到 {OUTPUT_JSON}")

with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Markdown 报告已保存到 {OUTPUT_MD}")

print("\n local_judge.py 全部完成")
