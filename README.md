# FinPilot — A股投资实战助手

> 基于 Qwen2.5-3B 的两阶段对齐微调项目：SFT 监督微调 → DPO 偏好对齐。  
> 在 50 题独立测试集上，DPO 模型对 SFT 基线的 GPT-4o 偏好胜率达 **74%**（+48pp）。

---

## 核心结果

| 指标 | SFT only | SFT + DPO | 变化 |
|------|:--------:|:---------:|:----:|
| GPT-4o 偏好胜率 | 26% | **74%** | +48pp |
| 含具体数字比例 | 100% | 100% | — |
| 含套话比例 | 14% | **4%** | −10pp |
| 回声开头比例 | 40% | **30%** | −10pp |
| 平均回答长度 | 134.7 字 | 136.9 字 | +2.2 字 |

> 评估方法：50 题独立测试集 × GPT-4o 裁判 × A/B 位置随机交换（消除位置偏见）

---

## 流程概览

```
原始金融数据 (3939条)
      │
      ▼
[数据清洗 + GPT-4o-mini 扩充]
      │
      ▼
SFT 数据集 (5000条 instruction/output)
      │
      ▼
┌─────────────────────────────────────┐
│  Stage 1: SFT 监督微调               │
│  Qwen2.5-3B + LoRA (rank=16)        │
│  lr=1e-4 / 3 epochs / FP16          │
│  6 组消融实验 → 选最优配置            │
└─────────────────────────────────────┘
      │
      ▼
[SFT adapter] ──merge_and_unload()──▶ merged base
      │
      ▼
DPO 偏好数据集 (1834组 chosen/rejected)
      │
      ▼
┌─────────────────────────────────────┐
│  Stage 2: DPO 偏好对齐               │
│  β=0.1 / lr=1e-5 / 1 epoch         │
│  sigmoid loss / 新 LoRA on merged   │
│  11 次消融实验（6 个维度）            │
└─────────────────────────────────────┘
      │
      ▼
[DPO adapter]
      │
      ▼
评估：50题测试集 × GPT-4o裁判 → 胜率 74%
```

---

## 项目结构

```
FinPilot/
├── data/
│   ├── sample_sft.json          # SFT 数据样例（20 条，完整集 5000 条未上传）
│   └── sample_dpo.json          # DPO 偏好对样例（20 组，完整集 1834 组未上传）
├── scripts/
│   ├── data/
│   │   ├── generate_finance_sft.py   # GPT-4o-mini 生成 SFT 数据
│   │   ├── generate_dpo_data.py      # 构造 chosen/rejected 偏好对
│   │   └── merge_finance_sft.py      # 合并多来源数据
│   ├── train/
│   │   ├── train_final.py            # SFT 最终训练
│   │   ├── train_dpo.py              # DPO 训练（含 11 次消融）
│   │   └── run_ablation.py           # SFT 消融实验（6 组）
│   └── eval/
│       ├── eval_dpo.py               # 生成 SFT/DPO 对比回答（AutoDL）
│       ├── generate_answers.py       # 独立生成回答文件
│       └── local_judge.py            # 本地 GPT-4o 裁判评估
├── demo/
│   └── demo_autodl.py               # Gradio 演示界面（AutoDL FP16）
├── results/
│   ├── dpo_results.json             # DPO 11 次消融完整指标
│   ├── ablation_results.json        # SFT 6 组消融结果
│   ├── eval_results.json            # GPT-4o 评估 50 个原始结果
│   ├── eval_report.md               # 评估报告
│   ├── dpo_summary.md               # DPO 消融汇总表
│   └── ablation_report.md           # SFT 消融报告
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 模型权重

LoRA adapter 文件约 120MB，未上传至本仓库。**训练脚本完全可复现**，按下方步骤即可重新训练。

如需直接使用预训练的 adapter，请通过 Issue 联系作者，或等待后续上传至 HuggingFace Hub。

- SFT adapter：`Qwen2.5-3B` + LoRA (rank=16, alpha=32, 全部线性层)
- DPO adapter：基于 merged SFT 模型 + 新 LoRA，β=0.1

---

## 环境配置

```bash
# 推荐 Python 3.12 + CUDA 12.8（本项目在 AutoDL RTX 5090 32GB 上训练）
pip install -r requirements.txt
```

> **注意 trl 版本**：本项目使用 `trl==1.0.0`，该版本的 `DPOConfig` API 与 0.x 版本不兼容，直接升级会报错。

---

## 复现步骤

### 1. 生成 SFT 数据

```bash
export OPENAI_API_KEY=sk-xxx
python scripts/data/generate_finance_sft.py   # 生成 ~5000 条
python scripts/data/merge_finance_sft.py       # 合并清洗
```

### 2. SFT 训练

```bash
# 修改脚本中的 MODEL_PATH 和 OUTPUT_DIR
python scripts/train/train_final.py
```

### 3. 生成 DPO 偏好数据

```bash
python scripts/data/generate_dpo_data.py      # 输出 chosen/rejected 对
```

### 4. DPO 训练

```bash
python scripts/train/train_dpo.py
# 默认运行全部 11 次消融，修改脚本顶部的 EXPERIMENTS 列表可单独运行
```

### 5. 评估

```bash
# 在 AutoDL 上生成对比回答
python scripts/eval/eval_dpo.py

# 下载 eval_answers.json 到本地后运行裁判
export OPENAI_API_KEY=sk-xxx
python scripts/eval/local_judge.py
```

---

## 消融实验结论

### SFT 阶段（6 组）

| 消融维度 | 最优配置 | 结论 |
|---------|---------|------|
| 量化精度 | FP16 | 4-bit 量化在 32GB 显存下无必要，FP16 训练更稳 |
| LoRA rank | rank=16 | rank=32 无显著提升，rank=8 略有下降 |
| target_modules | 全部线性层 | 仅 q/v 效果弱，全部线性层更充分 |
| 学习率 | 1e-4 | 5e-5 收敛慢，2e-4 不稳定 |
| epochs | 3 | 2 epochs 欠拟合，4 epochs 过拟合 |
| 数据量 | 5000 条 | 3000 条指标明显低于 5000 条 |

### DPO 阶段（6 维度 11 次）

| 消融维度 | 最优配置 | 结论 |
|---------|---------|------|
| β 参数 | **0.1** | β=0.5 过度约束，β=0.05 偏好漂移 |
| 学习率 | **1e-5** | 5e-6 学习不足，5e-5 发散 |
| epoch 数 | **1** | 2+ epochs 出现过拟合 |
| 长度差消融 | 等长数据≈原始 | DPO 学的是内容质量而非长度 |
| loss 类型 | **sigmoid** | IPO loss 效果相当但不稳定 |
| 噪声鲁棒性 | 10% 噪声可接受 | 30% 噪声性能明显下降 |

---

## 数据格式

**SFT 数据**（`data/sample_sft.json`）：
```json
{
  "instruction": "主力吸筹时 K 线有什么特征？",
  "input": "",
  "output": "主力吸筹阶段..."
}
```

**DPO 数据**（`data/sample_dpo.json`）：
```json
{
  "prompt": "主力吸筹时 K 线有什么特征？",
  "chosen": "主力吸筹通常出现在低位横盘阶段...",
  "rejected": "关于主力吸筹的 K 线特征，这是一个很好的问题..."
}
```

---

## 技术栈

| 组件 | 版本 |
|------|------|
| 基座模型 | Qwen2.5-3B |
| PyTorch | 2.8.0 + CUDA 12.8 |
| transformers | 5.5.0 |
| trl | 1.0.0 |
| peft | 0.18.1 |
| 数据生成 | OpenAI GPT-4o-mini |
| 评估裁判 | OpenAI GPT-4o |
| 演示界面 | Gradio |
| 训练硬件 | AutoDL RTX 5090 32GB |

---

## License

MIT
