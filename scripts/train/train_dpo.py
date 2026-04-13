# -*- coding: utf-8 -*-
"""
11 次消融实验：β / lr / epoch / 长度差 / loss类型 / 噪声比例
"""

import sys
import inspect

import trl
print(f"trl version: {trl.__version__}")
import transformers
print(f"transformers version: {transformers.__version__}")

from trl import DPOTrainer, DPOConfig

print("\n=== DPOConfig 可用参数 ===")
print(inspect.signature(DPOConfig.__init__))
print("\n=== DPOTrainer 可用参数 ===")
print(inspect.signature(DPOTrainer.__init__))
sys.stdout.flush()

import json
import os
import random
import time

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B"
    "/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
)
SFT_ADAPTER_PATH = "/root/autodl-tmp/final_model"
DPO_DATA_PATH    = "/root/autodl-tmp/data/finance_dpo_data.json"
DPO_MODELS_DIR   = "./dpo_models"
RESULTS_FILE     = "dpo_results.json"

os.makedirs(DPO_MODELS_DIR, exist_ok=True)


print("\n加载 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


# 模型加载
def load_base_with_sft_merged():
    print("  加载基座模型 ...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("  加载 SFT LoRA adapter ...")
    model = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH)
    print("  merge_and_unload() ...")
    model = model.merge_and_unload()
    return model


random.seed(42)

print("\n加载原始 DPO 数据 ...")
with open(DPO_DATA_PATH, encoding="utf-8") as f:
    raw_data = json.load(f)
print(f"原始数据条数: {len(raw_data)}")

# 构建原始数据
dataset_original = raw_data  

# 构建等长数据
print("构造变体 B（等长 rejected）...")

all_sentences = []
for d in raw_data:
    sents = [s.strip() for s in d["rejected"].split("。") if len(s.strip()) > 5]
    all_sentences.extend(sents)

random.seed(42)
equal_data = []
for d in raw_data:
    chosen_len = len(d["chosen"])
    rejected   = d["rejected"]

    while len(rejected) < chosen_len:
        extra    = random.choice(all_sentences)
        rejected = rejected + "。" + extra

    rejected = rejected[:chosen_len]
    last_period = rejected.rfind("。")
    if last_period > chosen_len * 0.8:
        rejected = rejected[: last_period + 1]

    equal_data.append(
        {"prompt": d["prompt"], "chosen": d["chosen"], "rejected": rejected}
    )
print(f"  等长变体条数: {len(equal_data)}")

# 噪声数据
print("构造变体 C/D（噪声数据）...")
random.seed(42)
all_noise_indices = random.sample(range(len(raw_data)), int(len(raw_data) * 0.30))
noise_30_indices  = set(all_noise_indices)
noise_10_indices  = set(all_noise_indices[: int(len(raw_data) * 0.10)])

noise_10_data = [dict(d) for d in raw_data]
for i in noise_10_indices:
    noise_10_data[i]["chosen"], noise_10_data[i]["rejected"] = (
        noise_10_data[i]["rejected"],
        noise_10_data[i]["chosen"],
    )

noise_30_data = [dict(d) for d in raw_data]
for i in noise_30_indices:
    noise_30_data[i]["chosen"], noise_30_data[i]["rejected"] = (
        noise_30_data[i]["rejected"],
        noise_30_data[i]["chosen"],
    )

print(f"  噪声 10% 条数: {len(noise_10_data)}，翻转数: {len(noise_10_indices)}")
print(f"  噪声 30% 条数: {len(noise_30_data)}，翻转数: {len(noise_30_indices)}")


DATA_VARIANTS = {
    "original":     dataset_original,
    "equal_length": equal_data,
    "noise_10":     noise_10_data,
    "noise_30":     noise_30_data,
}

def load_dataset_variant(variant_name: str) -> Dataset:
    data = DATA_VARIANTS[variant_name]
    return Dataset.from_list(
        [{"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]} for d in data]
    )


# LoRA 配置
NEW_LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)


# 11 次实验列表
BASELINE = {
    "beta":         0.1,
    "lr":           1e-5,
    "epochs":       1,
    "loss_type":    "sigmoid",
    "data_variant": "original",
}

experiments = [
    # Baseline
    {"name": "baseline",          "ablation": "shared", **BASELINE},
    # β 消融
    {"name": "beta_0.05",         "ablation": "beta",   **{**BASELINE, "beta": 0.05}},
    {"name": "beta_0.3",          "ablation": "beta",   **{**BASELINE, "beta": 0.3}},
    # 学习率消融
    {"name": "lr_5e-6",           "ablation": "lr",     **{**BASELINE, "lr": 5e-6}},
    {"name": "lr_5e-5",           "ablation": "lr",     **{**BASELINE, "lr": 5e-5}},
    # epoch 消融
    {"name": "epoch_2",           "ablation": "epoch",  **{**BASELINE, "epochs": 2}},
    {"name": "epoch_3",           "ablation": "epoch",  **{**BASELINE, "epochs": 3}},
    # 长度差消融
    {"name": "data_equal_length", "ablation": "length", **{**BASELINE, "data_variant": "equal_length"}},
    # loss 类型消融
    {"name": "loss_ipo",          "ablation": "loss",   **{**BASELINE, "loss_type": "ipo"}},
    # 噪声消融
    {"name": "noise_10",          "ablation": "noise",  **{**BASELINE, "data_variant": "noise_10"}},
    {"name": "noise_30",          "ablation": "noise",  **{**BASELINE, "data_variant": "noise_30"}},
]

assert len(experiments) == 11, f"预期 11 次实验，实际 {len(experiments)}"


_dpo_config_params = set(inspect.signature(DPOConfig.__init__).parameters.keys())

def build_dpo_config(exp: dict, output_dir: str, batch_size: int, grad_accum: int) -> DPOConfig:
    base_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=exp["lr"],
        num_train_epochs=exp["epochs"],
        beta=exp["beta"],
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    for key, val in [
        ("loss_type",        exp["loss_type"]),
        ("max_length",       1024),
        ("max_prompt_length", 256),
    ]:
        if key in _dpo_config_params:
            base_kwargs[key] = val
        else:
            print(f"    DPOConfig 不支持参数 '{key}'，已跳过")

    return DPOConfig(**base_kwargs)


# 带 OOM 重试
def run_training(exp: dict) -> dict:
    """
    执行一次 DPO 训练，OOM 时自动降低 batch size 重试一次。
    返回 results dict。
    """
    dataset = load_dataset_variant(exp["data_variant"])

    for attempt, (bs, ga) in enumerate([(2, 8), (1, 16)], start=1):
        try:
            print(f"  尝试 #{attempt}：batch_size={bs}, grad_accum={ga}")
            model = load_base_with_sft_merged()
            model = get_peft_model(model, NEW_LORA_CONFIG)

            save_path = os.path.join(DPO_MODELS_DIR, exp["name"])
            dpo_config = build_dpo_config(exp, save_path, bs, ga)

            # 检测 DPOTrainer 是否接受 tokenizer 参数
            trainer_params = set(inspect.signature(DPOTrainer.__init__).parameters.keys())
            trainer_kwargs = dict(
                model=model,
                ref_model=None,
                args=dpo_config,
                train_dataset=dataset,
            )
            if "tokenizer" in trainer_params:
                trainer_kwargs["tokenizer"] = tokenizer
            elif "processing_class" in trainer_params:
                trainer_kwargs["processing_class"] = tokenizer
            else:
                trainer_kwargs["tokenizer"] = tokenizer  # fallback

            trainer = DPOTrainer(**trainer_kwargs)

            start = time.time()
            train_result = trainer.train()
            elapsed = time.time() - start

            # 提取指标
            train_loss = train_result.metrics.get("train_loss", None)
            log_history = trainer.state.log_history
            rewards_acc_list = [
                log["rewards/accuracies"]
                for log in log_history
                if "rewards/accuracies" in log
            ]
            final_reward_acc = rewards_acc_list[-1] if rewards_acc_list else None

            # 保存 adapter
            trainer.save_model(save_path)
            print(f"   adapter 已保存到 {save_path}")

            del model, trainer
            torch.cuda.empty_cache()

            return {
                "ablation":        exp["ablation"],
                "config":          {
                    "beta":         exp["beta"],
                    "lr":           exp["lr"],
                    "epochs":       exp["epochs"],
                    "loss_type":    exp["loss_type"],
                    "data_variant": exp["data_variant"],
                },
                "train_loss":      train_loss,
                "reward_accuracy": final_reward_acc,
                "elapsed_minutes": round(elapsed / 60, 2),
                "model_path":      save_path,
                "batch_size_used": bs,
            }

        except torch.cuda.OutOfMemoryError as oom:
            print(f"   OOM（尝试 #{attempt}）：{oom}")
            try:
                del model, trainer
            except Exception:
                pass
            torch.cuda.empty_cache()
            if attempt == 2:
                raise RuntimeError(f"实验 {exp['name']} 两次尝试均 OOM，放弃") from oom
            print("  降低 batch size 重试 ...")


# 10. 主训练循环
results: dict = {}
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)
    print(f"\n发现已有结果文件，已完成 {len(results)} / 11 组")

for exp in experiments:
    if exp["name"] in results:
        print(f"  跳过已完成: {exp['name']}")
        continue

    print(f"\n{'='*60}")
    print(f"开始实验: {exp['name']}  ({experiments.index(exp)+1}/11)")
    print(f"β={exp['beta']}  lr={exp['lr']}  epochs={exp['epochs']}  "
          f"loss={exp['loss_type']}  data={exp['data_variant']}")
    print(f"{'='*60}")
    sys.stdout.flush()

    result = run_training(exp)
    results[exp["name"]] = result

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    rl = result["reward_accuracy"]
    tl = result["train_loss"]
    tl_str = f"{tl:.4f}" if tl is not None else "N/A"
    rl_str = f"{rl:.4f}" if rl is not None else "N/A"
    print(f" {exp['name']}: loss={tl_str}, reward_acc={rl_str}, time={result['elapsed_minutes']:.1f}min")

    # 基线完成后暂停 5 秒，方便检查显存
    if exp["name"] == "baseline":
        print("\n  基线训练完成，5 秒后继续...")
        sys.stdout.flush()
        time.sleep(5)

print("\n\n所有 11 次实验完成！")


# 汇总
def fmt(val, fmt_str=".4f"):
    return f"{val:{fmt_str}}" if val is not None else "N/A"

def row(name, r):
    return (
        f"| {name} | {fmt(r['config']['beta'])} | {r['config']['lr']:.0e} "
        f"| {r['config']['epochs']} | {r['config']['loss_type']} "
        f"| {r['config']['data_variant']} "
        f"| {fmt(r['train_loss'])} | {fmt(r['reward_accuracy'])} "
        f"| {r['elapsed_minutes']:.1f} |"
    )

def ablation_table(title, names_in_order, baseline_name="baseline"):
    lines = [f"\n### {title}"]
    lines.append("| 实验名 | β | lr | epochs | loss | data | train_loss | reward_acc | 耗时(min) |")
    lines.append("|--------|---|----|----|------|------|-----------|-----------|----------|")
    # baseline 排第一
    if baseline_name in results:
        lines.append(row(baseline_name + " ←base", results[baseline_name]))
    for n in names_in_order:
        if n != baseline_name and n in results:
            lines.append(row(n, results[n]))
    return "\n".join(lines)

summary = "\n## DPO 消融实验结果\n"
summary += ablation_table("1. β 参数消融", ["beta_0.05", "baseline", "beta_0.3"])
summary += ablation_table("2. 学习率消融", ["lr_5e-6", "baseline", "lr_5e-5"])
summary += ablation_table("3. Epoch 消融", ["baseline", "epoch_2", "epoch_3"])
summary += ablation_table("4. 长度差消融", ["baseline", "data_equal_length"])
summary += ablation_table("5. Loss 类型消融", ["baseline", "loss_ipo"])
summary += ablation_table("6. 噪声比例消融", ["baseline", "noise_10", "noise_30"])

# Top-3 by reward_accuracy
ranked = sorted(
    [(n, r) for n, r in results.items() if r["reward_accuracy"] is not None],
    key=lambda x: x[1]["reward_accuracy"],
    reverse=True,
)
summary += "\n\n### 最优配置（按 reward_accuracy 排序 Top 3）\n"
for i, (n, r) in enumerate(ranked[:3], 1):
    summary += f"{i}. **{n}** — reward_acc={fmt(r['reward_accuracy'])}, loss={fmt(r['train_loss'])}\n"

print(summary)

with open("dpo_summary.md", "w", encoding="utf-8") as f:
    f.write(summary)
print("汇总表已保存到 dpo_summary.md")


# 推理对比（最优配置 vs SFT）
if not ranked:
    print("\n无有效 reward_accuracy，跳过推理对比")
    sys.exit(0)

best_name, best_result = ranked[0]
best_adapter_path = best_result["model_path"]
print(f"\n最优配置: {best_name}，adapter 路径: {best_adapter_path}")

test_questions = [
    "MACD底背离了但股价还在跌，怎么回事？",
    "一只股票连续三天缩量下跌，是洗盘还是出货？",
    "ROE连续三年大于20%的股票就值得买吗？",
    "北向资金连续流出一周，要不要清仓？",
    "定投沪深300亏了两年了，还要继续吗？",
]

@torch.no_grad()
def generate(model, question: str, max_new_tokens: int = 300) -> str:
    prompt = f"### 指令:\n{question}\n\n### 回答:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

print("\n加载 SFT-only 模型 ...")
sft_base = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
sft_model = PeftModel.from_pretrained(sft_base, SFT_ADAPTER_PATH)

print(f"加载最优 DPO 模型（{best_name}）...")
dpo_base   = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
dpo_merged = PeftModel.from_pretrained(dpo_base, SFT_ADAPTER_PATH).merge_and_unload()
dpo_model  = PeftModel.from_pretrained(dpo_merged, best_adapter_path)

print("\n\n## 推理对比（SFT-only vs SFT+DPO）")
print(f"最优配置: {best_name}\n")

for q in test_questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    sft_ans = generate(sft_model, q)
    dpo_ans = generate(dpo_model, q)
    print(f"\n[SFT-only]\n{sft_ans}")
    print(f"\n[SFT+DPO ({best_name})]\n{dpo_ans}")

del sft_model, dpo_model
torch.cuda.empty_cache()
print("\n\n train_dpo.py 全部完成")
