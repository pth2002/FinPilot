# A股投资实战助手 — Qwen2.5-3B SFT + DPO 微调

> 基于 Qwen2.5-3B 的两阶段对齐微调项目，完整实现 SFT 监督微调 + DPO 偏好对齐，在 50 题独立测试集上对 SFT 基线的偏好胜率达到 **74%**（GPT-4o 评判）。

## 📊 核心结果

| 指标 | SFT only | SFT + DPO | 变化 |
|------|---------|-----------|------|
| GPT-4o 偏好胜率 | 26% | **74%** | +48pp |
| 含具体数字比例 | 100% | 100% | — |
| 含套话比例 | 14% | 4% | −10pp |
| 回声开头比例 | 40% | 30% | −10pp |
| 平均回答长度 | 134.7 字 | 136.9 字 | +2.2 字 |

## 🎯 项目亮点

- **完整的两阶段对齐流程**：SFT 监督微调 → DPO 偏好对齐
- **系统性消融实验**：SFT 阶段 6 组消融，DPO 阶段 6 维度共 11 次训练
- **严谨的评估方法**：GPT-4o 偏好胜率 + A/B 位置随机交换避免位置偏见 + 4 个辅助特征指标
- **创新消融设计**：长度差消融（验证 DPO 是否学到长度而非内容）、噪声鲁棒性消融（0%/10%/30% 标注错误）

## 📁 项目结构

```
├── data/
│   ├── sample_sft.json          # SFT 数据样例（20 条）
│   └── sample_dpo.json          # DPO 偏好对样例（20 组）
├── generate_finance_sft.py      # SFT 数据生成
├── generate_dpo_data.py         # DPO 偏好数据生成
├── run_ablation.py              # SFT 消融实验
├── train_final.py               # SFT 最终训练
├── train_dpo.py                 # DPO 训练（含消融）
├── eval_dpo.py                  # DPO 评估（AutoDL 版）
├── local_judge.py               # 本地 GPT-4o 裁判
├── generate_answers.py          # 分离评估生成
├── merge_finance_sft.py         # 数据合并脚本
├── demo_autodl.py               # Gradio 演示界面
├── dpo_results.json             # DPO 消融完整结果（11 次）
├── ablation_results.json        # SFT 消融结果（6 组）
├── eval_results.json            # 评估原始数据（50 题）
├── eval_report.md               # 评估报告
├── dpo_summary.md               # DPO 消融汇总表
└── README.md
```

## 🔬 方法论

### SFT 阶段

- **基座模型**：Qwen2.5-3B (Base)
- **训练方法**：LoRA (rank=16，target_modules=全部线性层)
- **训练参数**：FP16，lr=1e-4，3 epochs，batch_size=4×grad_accum=4
- **数据集**：5000 条高质量A股问答（从 3939 条原始数据清洗扩充）

### DPO 阶段

- **偏好数据**：1834 组（chosen 平均 136 字，含具体数字；rejected 平均 80 字，套话型）
- **最优配置**：β=0.1，lr=1e-5，1 epoch，sigmoid loss
- **模型加载**：base → SFT LoRA → merge_and_unload() → 新 DPO LoRA
- **消融实验**：β 参数 / 学习率 / epoch / 长度差 / DPO vs IPO / 噪声鲁棒性

### 评估方法

- **主指标**：GPT-4o 偏好胜率，50 题独立测试集
- **位置偏见控制**：A/B 位置随机交换
- **辅助指标**：含数字比例、含套话比例、回声开头比例、平均长度

## 🧪 关键实验结论

1. **社区推荐的 DPO 默认配置最优**：β=0.1，lr=1e-5，1 epoch 就是最佳组合，其他超参数没有显著提升
2. **DPO 学到的是内容质量而非长度**：长度差消融中等长数据和原始数据效果相当
3. **DPO 对 10% 以内的标注噪声鲁棒**：30% 噪声时性能明显下降
4. **含数字比例已被 SFT 训到 100% 上限**，DPO 的提升体现在"含套话比例" −10pp 和"回声开头" −10pp

## 🛠 技术栈

- **模型**：Qwen2.5-3B
- **训练框架**：PyTorch、transformers、peft、trl
- **数据生成**：OpenAI GPT-4o-mini
- **评估**：OpenAI GPT-4o
- **部署**：Gradio，AutoDL (RTX 5090)

## 📝 数据样例

完整数据集未上传（5000 条 SFT + 1834 组 DPO），详见 `data/sample_*.json`。

## ⚠️ 注意事项

- 模型权重（LoRA adapter）未上传（约 120MB × 2），需要自行训练
- 完整数据集未上传，仓库只提供 20 条样例
- 脚本中的路径需要根据实际环境调整
- 评估脚本依赖 OpenAI API，需要配置 `OPENAI_API_KEY`

## 📄 License

MIT
