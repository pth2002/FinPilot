#!/usr/bin/env python3
"""
train_final.py — Qwen2.5-3B + LoRA 最终训练脚本
"""
import gc
import importlib.metadata
import json
import math
import os
import random
import sys
import time

import torch

def print_env():
    print("=" * 60)
    print("环境信息")
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  PyTorch    : {torch.__version__}")
    print(f"  CUDA       : {torch.version.cuda}")
    for pkg in ("transformers", "peft", "trl", "datasets", "accelerate"):
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "NOT INSTALLED"
        print(f"  {pkg:<12}: {ver}")
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"  GPU        : {dev.name}  {total / 1024**3:.1f} GB total")
        print(f"  VRAM free  : {free / 1024**3:.1f} GB")
    else:
        print("  GPU        : NOT AVAILABLE")
    print("=" * 60)

print_env()

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


MODEL_PATH = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-3B/snapshots/"
    "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
)
DATA_PATH  = "/root/autodl-tmp/data/finance_sft_final.json"
OUTPUT_DIR = "./final_model"
LOG_DIR    = "./logs"

# LoRA
LORA_RANK       = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]

# 训练
LEARNING_RATE  = 1e-4
NUM_EPOCHS     = 3
PER_DEVICE_BATCH  = 8
GRAD_ACCUM        = 4       
MAX_SEQ_LEN       = 1024
WARMUP_RATIO      = 0.05
EVAL_RATIO        = 0.1
SEED              = 42

TEST_QUESTIONS = [
    "MACD底背离了但股价还在跌，是什么情况？",
    "一只股票ROE很高但负债率也很高，还能买吗？",
    "主力拉升后快速缩量横盘，是出货信号吗？",
    "定投沪深300两年了还亏着，该止损还是继续？",
    "集合竞价阶段突然大幅拉高，能不能追进去？",
]


# 数据
def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(SEED)
    rng.shuffle(data)
    split = int(len(data) * (1 - EVAL_RATIO))
    print(f"数据集: 共 {len(data)} 条 → train={split}, eval={len(data)-split}")
    return data[:split], data[split:]


def build_datasets(train_raw, eval_raw, tokenizer):
    """tokenize + label mask（prompt 部分 label=-100，只对 response 算 loss）"""

    def process_batch(examples):
        all_input_ids, all_attn_mask, all_labels = [], [], []

        for inst, inp, out in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            inp = inp.strip() if inp else ""
            if inp:
                prompt = (
                    f"### Instruction:\n{inst}\n\n"
                    f"### Input:\n{inp}\n\n"
                    f"### Response:\n"
                )
            else:
                prompt = f"### Instruction:\n{inst}\n\n### Response:\n"

            full_text = prompt + out + tokenizer.eos_token

            full_enc   = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LEN)
            prompt_len = len(
                tokenizer(prompt, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
            )

            ids    = full_enc["input_ids"]
            labels = [-100] * min(prompt_len, len(ids)) + ids[min(prompt_len, len(ids)):]

            all_input_ids.append(ids)
            all_attn_mask.append(full_enc["attention_mask"])
            all_labels.append(labels)

        return {
            "input_ids":      all_input_ids,
            "attention_mask": all_attn_mask,
            "labels":         all_labels,
        }

    def to_dataset(raw):
        ds = Dataset.from_list(raw)
        return ds.map(
            process_batch,
            batched=True,
            batch_size=512,
            remove_columns=ds.column_names,
            desc="Tokenizing",
        )

    return to_dataset(train_raw), to_dataset(eval_raw)



# 模型
def build_model():
    print(f"\nLoading model: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model



# 训练
def train(model, train_ds, eval_ds, tokenizer):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        fp16=True,
        bf16=False,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_dir=LOG_DIR,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    print("\n开始训练...")
    t0 = time.time()
    trainer.train()
    elapsed = (time.time() - t0) / 60

    eval_result = trainer.evaluate()
    eval_loss = eval_result["eval_loss"]
    print(f"\n训练完成: {elapsed:.1f} min")
    print(f"eval_loss = {eval_loss:.4f},  eval_ppl = {math.exp(eval_loss):.2f}")

    # 保存 LoRA adapter + tokenizer
    print(f"\n保存 LoRA adapter 到 {OUTPUT_DIR} ...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    summary = {
        "model_path":      MODEL_PATH,
        "lora_rank":       LORA_RANK,
        "target_modules":  TARGET_MODULES,
        "learning_rate":   LEARNING_RATE,
        "num_epochs":      NUM_EPOCHS,
        "batch_size":      PER_DEVICE_BATCH,
        "grad_accum":      GRAD_ACCUM,
        "max_seq_len":     MAX_SEQ_LEN,
        "train_samples":   len(train_ds),
        "eval_samples":    len(eval_ds),
        "eval_loss":       round(eval_loss, 4),
        "eval_ppl":        round(math.exp(eval_loss), 2),
        "train_time_min":  round(elapsed, 1),
        "finished_at":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return trainer



# 推理测试（训练显存释放后重新加载）
def inference_test():
    """释放训练显存后，重新加载 base model + LoRA adapter 进行推理测试。"""
    from peft import PeftModel

    print("\n" + "=" * 60)
    print("推理测试")
    print("=" * 60)

    # 显存状态
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"加载前 VRAM: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")

    print(f"Loading base model for inference ...")
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.eval()

    for i, question in enumerate(TEST_QUESTIONS, 1):
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        answer  = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        print(f"\nQ{i}: {question}")
        print(f"A{i}: {answer}")
        print(f"    [{len(answer)}字]")

    print("\n" + "=" * 60)

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()



# 主流程
def main():
    # tokenizer
    print(f"\nLoading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 数据
    train_raw, eval_raw = load_data()
    train_ds, eval_ds   = build_datasets(train_raw, eval_raw, tokenizer)

    # 模型 + 训练
    model   = build_model()
    trainer = train(model, train_ds, eval_ds, tokenizer)

    # 释放训练显存
    print("\n释放训练显存...")
    del trainer, model, train_ds, eval_ds, train_raw, eval_raw
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"释放后 VRAM: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")

    # 推理测试
    inference_test()

    print(f"\n全部完成。LoRA adapter 保存于: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
