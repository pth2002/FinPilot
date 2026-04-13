#!/usr/bin/env python3

import gc
import json
import math
import os
import random
import time
import traceback
from pathlib import Path

import torch
from datasets import Dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


MODEL_NAME   = "Qwen/Qwen2.5-3B"
DATA_PATH    = "./data/finance_sft_final.json"
RESULTS_FILE = "ablation_results.json"
OUTPUT_DIR   = "ablation_runs"
MAX_SEQ_LEN  = 1024
BATCH_SIZE   = 4
GRAD_ACCUM   = 4
EVAL_RATIO   = 0.1
SEED         = 42

fp16_oom_detected = False


# 消融实验定义
EXPERIMENTS = [
    # 量化消融
    dict(name="quant_fp16",  group="量化消融",           label="FP16",
         rank=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
         force_4bit=False, n_samples=5000, lr=1e-4, epochs=3),
    dict(name="quant_4bit",  group="量化消融",           label="4bit QLoRA",
         rank=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
         force_4bit=True,  n_samples=5000, lr=1e-4, epochs=3),

    # rank 消融
    dict(name="rank_4",      group="rank消融",           label="rank=4",
         rank=4,  lora_alpha=8,   target_modules=["q_proj", "v_proj"],
         force_4bit=False, n_samples=5000, lr=1e-4, epochs=3),
    dict(name="rank_16",     group="rank消融",           label="rank=16",
         rank=16, lora_alpha=32,  target_modules=["q_proj", "v_proj"],
         force_4bit=False, n_samples=5000, lr=1e-4, epochs=3),
    dict(name="rank_64",     group="rank消融",           label="rank=64",
         rank=64, lora_alpha=128, target_modules=["q_proj", "v_proj"],
         force_4bit=False, n_samples=5000, lr=1e-4, epochs=3),

    # target_modules 消融
    dict(name="target_qv",   group="target_modules消融", label="q_proj+v_proj",
         rank=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
         force_4bit=False, n_samples=5000, lr=1e-4, epochs=3),
    dict(name="target_all",  group="target_modules消融", label="全部线性层",
         rank=16, lora_alpha=32,
         target_modules=["q_proj","k_proj","v_proj","o_proj",
                         "gate_proj","up_proj","down_proj"],
         force_4bit=False, n_samples=5000, lr=1e-4, epochs=3),

    # 数据量消融
    dict(name="data_5000",   group="数据量消融",         label="全量5000条",
         rank=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
         force_4bit=False, n_samples=5000, lr=1e-4, epochs=3),
    dict(name="data_1500",   group="数据量消融",         label="随机抽样1500条",
         rank=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
         force_4bit=False, n_samples=1500, lr=1e-4, epochs=3),

    # 学习率消融
    dict(name="lr_1e3",      group="学习率消融",         label="lr=1e-3",
         rank=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
         force_4bit=False, n_samples=5000, lr=1e-3, epochs=3),
    dict(name="lr_1e4",      group="学习率消融",         label="lr=1e-4",
         rank=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
         force_4bit=False, n_samples=5000, lr=1e-4, epochs=3),
    dict(name="lr_5e5",      group="学习率消融",         label="lr=5e-5",
         rank=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
         force_4bit=False, n_samples=5000, lr=5e-5, epochs=3),

    # 训练轮数消融
    dict(name="epoch_1",     group="训练轮数消融",       label="1 epoch",
         rank=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
         force_4bit=False, n_samples=5000, lr=1e-4, epochs=1),
    dict(name="epoch_3",     group="训练轮数消融",       label="3 epochs",
         rank=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
         force_4bit=False, n_samples=5000, lr=1e-4, epochs=3),
    dict(name="epoch_5",     group="训练轮数消融",       label="5 epochs",
         rank=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
         force_4bit=False, n_samples=5000, lr=1e-4, epochs=5),
]


def load_results() -> dict:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results: dict):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)



# 数据加载
def load_and_split(n_samples: int):
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)

    rng = random.Random(SEED)
    rng.shuffle(data)

    if n_samples < len(data):
        data = data[:n_samples]

    split = max(1, int(len(data) * (1 - EVAL_RATIO)))
    return data[:split], data[split:]


def build_datasets(train_raw, eval_raw, tokenizer) -> tuple[Dataset, Dataset]:
    def process(examples):
        all_input_ids, all_attention_mask, all_labels = [], [], []

        for inst, inp, out in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            if inp and inp.strip():
                prompt = (
                    f"### Instruction:\n{inst}\n\n"
                    f"### Input:\n{inp}\n\n"
                    f"### Response:\n"
                )
            else:
                prompt = f"### Instruction:\n{inst}\n\n### Response:\n"

            full_text = prompt + out + tokenizer.eos_token

            full_enc   = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LEN)
            prompt_enc = tokenizer(prompt,    truncation=True, max_length=MAX_SEQ_LEN)

            input_ids  = full_enc["input_ids"]
            prompt_len = min(len(prompt_enc["input_ids"]), len(input_ids))
            labels     = [-100] * prompt_len + input_ids[prompt_len:]

            all_input_ids.append(input_ids)
            all_attention_mask.append(full_enc["attention_mask"])
            all_labels.append(labels)

        return {
            "input_ids":      all_input_ids,
            "attention_mask": all_attention_mask,
            "labels":         all_labels,
        }

    def to_hf(raw):
        ds = Dataset.from_list(raw)
        return ds.map(
            process,
            batched=True,
            batch_size=256,
            remove_columns=ds.column_names,
            desc="Tokenizing",
        )

    return to_hf(train_raw), to_hf(eval_raw)



# 模型加载
def load_model(use_4bit: bool):
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.enable_input_require_grads()

    model.config.use_cache = False
    return model


def free_memory(model=None):
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()



# 单次实验
def run_one(exp: dict, tokenizer) -> dict:
    global fp16_oom_detected

    
    use_4bit = exp["force_4bit"] or fp16_oom_detected

    print(f"\n{'='*60}")
    print(f"[{exp['group']}] {exp['label']}")
    print(f"  rank={exp['rank']}, lr={exp['lr']}, epochs={exp['epochs']}")
    print(f"  target_modules={exp['target_modules']}")
    print(f"  n_samples={exp['n_samples']}, 4bit={use_4bit}")
    print(f"{'='*60}")

    # 数据
    train_raw, eval_raw = load_and_split(exp["n_samples"])
    print(f"  train={len(train_raw)}, eval={len(eval_raw)}")
    train_ds, eval_ds = build_datasets(train_raw, eval_raw, tokenizer)

    # 模型
    model = load_model(use_4bit)

    lora_cfg = LoraConfig(
        r=exp["rank"],
        lora_alpha=exp["lora_alpha"],
        target_modules=exp["target_modules"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 参数
    run_dir = os.path.join(OUTPUT_DIR, exp["name"])
    os.makedirs(run_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=exp["epochs"],
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=exp["lr"],
        fp16=(not use_4bit),
        bf16=False,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # 训练
    t0 = time.time()
    train_output = trainer.train()
    elapsed_min = (time.time() - t0) / 60

    train_loss = float("nan")
    for entry in reversed(trainer.state.log_history):
        if "loss" in entry and "eval_loss" not in entry:
            train_loss = entry["loss"]
            break

    eval_result = trainer.evaluate()
    eval_loss = eval_result["eval_loss"]
    eval_ppl  = math.exp(min(eval_loss, 20)) 

    result = {
        "group":          exp["group"],
        "label":          exp["label"],
        "use_4bit":       use_4bit,
        "train_loss":     round(train_loss, 4),
        "eval_loss":      round(eval_loss, 4),
        "eval_ppl":       round(eval_ppl, 2),
        "train_time_min": round(elapsed_min, 1),
        "finished_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    free_memory(model)
    return result



# 汇总
def print_summary(results: dict):
    # 按 group 排列
    group_order = [
        "量化消融", "rank消融", "target_modules消融",
        "数据量消融", "学习率消融", "训练轮数消融",
    ]

    print("\n\n" + "=" * 70)
    print("消融实验结果汇总")
    print("=" * 70)

    for group in group_order:
        rows = [r for r in results.values() if r.get("group") == group]
        if not rows:
            continue

        print(f"\n### {group}\n")
        print(f"| {'配置':<20} | {'train_loss':>10} | {'eval_loss':>9} | {'eval_ppl':>8} | {'时间(min)':>9} | {'量化':>6} |")
        print(f"|{'-'*22}|{'-'*12}|{'-'*11}|{'-'*10}|{'-'*11}|{'-'*8}|")
        for r in rows:
            quant = "4bit" if r.get("use_4bit") else "FP16"
            tl = f"{r['train_loss']:.4f}" if not math.isnan(r['train_loss']) else "  N/A"
            print(
                f"| {r['label']:<20} | {tl:>10} | {r['eval_loss']:>9.4f} "
                f"| {r['eval_ppl']:>8.2f} | {r['train_time_min']:>9.1f} | {quant:>6} |"
            )

    print()



# 主流程
def main():
    global fp16_oom_detected

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = load_results()

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    total = len(EXPERIMENTS)
    for i, exp in enumerate(EXPERIMENTS, 1):
        name = exp["name"]

        if name in results:
            print(f"[{i}/{total}] skip {name} (already done)")
            continue

        print(f"\n[{i}/{total}] Starting: {name}")

        try:
            result = run_one(exp, tokenizer)
            results[name] = result
            save_results(results)
            print(
                f"  -> eval_loss={result['eval_loss']:.4f}  "
                f"eval_ppl={result['eval_ppl']:.2f}  "
                f"time={result['train_time_min']:.1f}min"
            )

        except torch.cuda.OutOfMemoryError:
            free_memory()

            # FP16 OOM -> 切换全局 4bit
            if not exp["force_4bit"]:
                print(f"  [OOM] FP16 out of memory for {name}, switching to 4bit globally.")
                fp16_oom_detected = True

                # 4bit 重试
                try:
                    result = run_one(exp, tokenizer)
                    result["oom_fallback"] = True
                    results[name] = result
                    save_results(results)
                    print(
                        f"  -> (4bit fallback) eval_loss={result['eval_loss']:.4f}  "
                        f"eval_ppl={result['eval_ppl']:.2f}"
                    )
                except Exception as e2:
                    print(f"  [FAIL] 4bit retry also failed: {e2}")
                    results[name] = {"group": exp["group"], "label": exp["label"],
                                     "error": str(e2), "use_4bit": True}
                    save_results(results)
            else:
                print(f"  [FAIL] {name} OOM even with 4bit.")
                results[name] = {"group": exp["group"], "label": exp["label"],
                                 "error": "OOM", "use_4bit": True}
                save_results(results)

        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            results[name] = {"group": exp["group"], "label": exp["label"],
                             "error": str(e)}
            save_results(results)
            free_memory()

    print_summary(results)
    print(f"All results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
