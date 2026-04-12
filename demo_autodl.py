# -*- coding: utf-8 -*-
"""
demo_autodl.py — A股投资实战助手 AutoDL 演示版
硬件要求：RTX 5090 32GB（FP16/BF16，两模型合计约 14GB 显存）
用法：
    pip install gradio peft transformers accelerate -i https://mirrors.aliyun.com/pypi/simple/ -U
    python demo_autodl.py
"""

import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================
# 路径配置
# ============================================================
BASE_MODEL_PATH = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
SFT_ADAPTER_PATH = "/root/autodl-tmp/final_model"
DPO_ADAPTER_PATH = "/root/autodl-tmp/dpo_models/baseline"

# ============================================================
# 1. 加载 tokenizer
# ============================================================
print("加载 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print(f"[tokenizer] eos_token={tokenizer.eos_token!r}, eos_token_id={tokenizer.eos_token_id}")

# ============================================================
# 2. 加载 SFT 模型（基座 + SFT LoRA merge）
# ============================================================
print("加载 SFT 模型 ...")
sft_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
sft_model = PeftModel.from_pretrained(sft_base, SFT_ADAPTER_PATH)
sft_model = sft_model.merge_and_unload()
sft_model.eval()
print(f"[显存] SFT 加载后: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ============================================================
# 3. 加载 DPO 模型（基座 + SFT LoRA merge + DPO LoRA）
#    必须重新加载一份基座，不能复用 sft_model（已被 merge）
# ============================================================
print("加载 DPO 模型 ...")
dpo_base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
dpo_with_sft = PeftModel.from_pretrained(dpo_base, SFT_ADAPTER_PATH)
dpo_with_sft = dpo_with_sft.merge_and_unload()
dpo_model = PeftModel.from_pretrained(dpo_with_sft, DPO_ADAPTER_PATH)
dpo_model.eval()
print(f"[显存] DPO 加载后: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("[访问地址] AutoDL 控制台 → 自定义服务 → 选择端口 6006")

# ============================================================
# 4. 生成函数
# ============================================================
_STOP_TOKENS = ["### 指令", "\n### ", "### 回答", "\n###", "<|im_end|>", "<|endoftext|>"]

def generate_answer(model, question: str, max_new_tokens: int = 300) -> str:
    prompt = f"### 指令:\n{question.strip()}\n\n### 回答:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
        )

    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # 后处理：截断接龙（模型有时生成模板分隔符而非 EOS token）
    for stop in _STOP_TOKENS:
        if stop in answer:
            answer = answer.split(stop)[0].strip()
            break

    return answer.strip()

# ============================================================
# 5. Gradio 界面
# ============================================================
with gr.Blocks(title="A股投资实战助手 — SFT vs SFT+DPO 对比", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# A股投资实战助手")
    gr.Markdown(
        "**项目**：Qwen2.5-3B + LoRA SFT + DPO 偏好对齐 &nbsp;|&nbsp; "
        "**评估**：DPO 胜率 74%（GPT-4o 评判）"
    )

    with gr.Row():
        question_input = gr.Textbox(
            label="请输入你的股票投资问题",
            placeholder="例如：北向资金连续流出一周，要不要清仓？",
            lines=2,
        )

    with gr.Row():
        submit_btn = gr.Button("生成回答", variant="primary", size="lg")
        clear_btn  = gr.Button("清空", size="lg")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### SFT only（仅监督微调）")
            sft_output = gr.Textbox(label="", lines=15, show_label=False)
        with gr.Column():
            gr.Markdown("### SFT + DPO（偏好对齐后）")
            dpo_output = gr.Textbox(label="", lines=15, show_label=False)

    gr.Examples(
        examples=[
            "北向资金连续流出一周，要不要清仓？",
            "MACD 底背离了但股价还在跌，是怎么回事？",
            "ROE 连续三年大于 20% 的股票就值得买吗？",
            "一只股票连续三天缩量下跌，是洗盘还是出货？",
            "定投沪深 300 亏了两年了，还要继续吗？",
            "股票回调到 20 日均线，但成交量还在放大，该接吗？",
        ],
        inputs=question_input,
        label="试试这些问题",
    )

    def on_submit(question):
        if not question.strip():
            return "", ""
        sft_ans = generate_answer(sft_model, question)
        dpo_ans = generate_answer(dpo_model, question)
        return sft_ans, dpo_ans

    submit_btn.click(on_submit, inputs=question_input, outputs=[sft_output, dpo_output])
    question_input.submit(on_submit, inputs=question_input, outputs=[sft_output, dpo_output])
    clear_btn.click(lambda: ("", "", ""), outputs=[question_input, sft_output, dpo_output])

demo.queue(default_concurrency_limit=1, max_size=10).launch(
    server_name="0.0.0.0",
    server_port=6006,
    share=False,
    show_error=True,
    ssr_mode=False,
)
