# -*- coding: utf-8 -*-
"""
generate_answers.py — 离线生成 SFT / DPO 回答，保存供后续 GPT-4o 裁判使用
不调用任何外部 API
"""

import json
import random

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 路径配置
# ============================================================
MODEL_PATH       = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B"
    "/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
)
SFT_ADAPTER_PATH = "/root/autodl-tmp/final_model"
DPO_ADAPTER_PATH = "/root/autodl-tmp/dpo_models/baseline"
SFT_DATA_PATH    = "/root/autodl-tmp/data/finance_sft_final.json"
OUTPUT_PATH      = "eval_answers.json"

# ============================================================
# Tokenizer
# ============================================================
print("加载 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 模型加载工具
# ============================================================
def load_sft_model():
    print("加载 SFT-only 模型 ...")
    base  = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH)
    model.eval()
    return model

def load_dpo_model():
    print(f"加载 DPO 模型（adapter: {DPO_ADAPTER_PATH}）...")
    base   = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH).merge_and_unload()
    model  = PeftModel.from_pretrained(merged, DPO_ADAPTER_PATH)
    model.eval()
    return model

# ============================================================
# 生成工具
# ============================================================
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

# ============================================================
# 测试集（seed=2024，抽 50 条）
# ============================================================
print("准备测试集 ...")
with open(SFT_DATA_PATH, encoding="utf-8") as f:
    sft_data = json.load(f)

random.seed(2024)
test_set       = random.sample(sft_data, 50)
test_questions = [d["instruction"] for d in test_set]
print(f"测试集大小: {len(test_questions)} 条")

# ============================================================
# 生成 SFT 回答
# ============================================================
print("\n生成 SFT-only 回答 ...")
sft_model   = load_sft_model()
sft_answers = []
for i, q in enumerate(test_questions, 1):
    ans = generate(sft_model, q)
    sft_answers.append(ans)
    print(f"  [{i:02d}/50] {len(ans)} 字")

del sft_model
torch.cuda.empty_cache()

# ============================================================
# 生成 DPO 回答
# ============================================================
print("\n生成 SFT+DPO 回答 ...")
dpo_model   = load_dpo_model()
dpo_answers = []
for i, q in enumerate(test_questions, 1):
    ans = generate(dpo_model, q)
    dpo_answers.append(ans)
    print(f"  [{i:02d}/50] {len(ans)} 字")

del dpo_model
torch.cuda.empty_cache()

# ============================================================
# 保存结果
# ============================================================
output = [
    {
        "index":      i,
        "question":   q,
        "sft_answer": sft,
        "dpo_answer": dpo,
    }
    for i, (q, sft, dpo) in enumerate(zip(test_questions, sft_answers, dpo_answers))
]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ 已保存 {len(output)} 条回答到 {OUTPUT_PATH}")
