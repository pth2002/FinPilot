
## DPO 消融实验结果

### 1. β 参数消融
| 实验名 | β | lr | epochs | loss | data | train_loss | reward_acc | 耗时(min) |
|--------|---|----|----|------|------|-----------|-----------|----------|
| baseline ←base | 0.1000 | 1e-05 | 1 | sigmoid | original | 0.1486 | 1.0000 | 5.4 |
| beta_0.05 | 0.0500 | 1e-05 | 1 | sigmoid | original | 0.2068 | 1.0000 | 5.4 |
| beta_0.3 | 0.3000 | 1e-05 | 1 | sigmoid | original | 0.0922 | 1.0000 | 5.4 |
### 2. 学习率消融
| 实验名 | β | lr | epochs | loss | data | train_loss | reward_acc | 耗时(min) |
|--------|---|----|----|------|------|-----------|-----------|----------|
| baseline ←base | 0.1000 | 1e-05 | 1 | sigmoid | original | 0.1486 | 1.0000 | 5.4 |
| lr_5e-6 | 0.1000 | 5e-06 | 1 | sigmoid | original | 0.2985 | 1.0000 | 5.4 |
| lr_5e-5 | 0.1000 | 5e-05 | 1 | sigmoid | original | 0.0474 | 1.0000 | 5.3 |
### 3. Epoch 消融
| 实验名 | β | lr | epochs | loss | data | train_loss | reward_acc | 耗时(min) |
|--------|---|----|----|------|------|-----------|-----------|----------|
| baseline ←base | 0.1000 | 1e-05 | 1 | sigmoid | original | 0.1486 | 1.0000 | 5.4 |
| epoch_2 | 0.1000 | 1e-05 | 2 | sigmoid | original | 0.0814 | 1.0000 | 10.8 |
| epoch_3 | 0.1000 | 1e-05 | 3 | sigmoid | original | 0.0599 | 1.0000 | 16.1 |
### 4. 长度差消融
| 实验名 | β | lr | epochs | loss | data | train_loss | reward_acc | 耗时(min) |
|--------|---|----|----|------|------|-----------|-----------|----------|
| baseline ←base | 0.1000 | 1e-05 | 1 | sigmoid | original | 0.1486 | 1.0000 | 5.4 |
| data_equal_length | 0.1000 | 1e-05 | 1 | sigmoid | equal_length | 0.1227 | 1.0000 | 5.4 |
### 5. Loss 类型消融
| 实验名 | β | lr | epochs | loss | data | train_loss | reward_acc | 耗时(min) |
|--------|---|----|----|------|------|-----------|-----------|----------|
| baseline ←base | 0.1000 | 1e-05 | 1 | sigmoid | original | 0.1486 | 1.0000 | 5.4 |
| loss_ipo | 0.1000 | 1e-05 | 1 | ipo | original | 12.5124 | 1.0000 | 5.4 |
### 6. 噪声比例消融
| 实验名 | β | lr | epochs | loss | data | train_loss | reward_acc | 耗时(min) |
|--------|---|----|----|------|------|-----------|-----------|----------|
| baseline ←base | 0.1000 | 1e-05 | 1 | sigmoid | original | 0.1486 | 1.0000 | 5.4 |
| noise_10 | 0.1000 | 1e-05 | 1 | sigmoid | noise_10 | 0.3977 | 0.9189 | 5.3 |
| noise_30 | 0.1000 | 1e-05 | 1 | sigmoid | noise_30 | 0.6242 | 0.7568 | 5.4 |

### 最优配置（按 reward_accuracy 排序 Top 3）
1. **baseline** — reward_acc=1.0000, loss=0.1486
2. **beta_0.05** — reward_acc=1.0000, loss=0.2068
3. **beta_0.3** — reward_acc=1.0000, loss=0.0922
