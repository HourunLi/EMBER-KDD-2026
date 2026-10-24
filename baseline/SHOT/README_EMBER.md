# SHOT-GCN：EMBER 四数据集复现

在本目录修改 `run.py` 顶部配置后运行：

```bash
python run.py
```

- `DEFAULTS`：数据集、运行次数、GPU、种子和源监督协议。默认四个数据集各 5 次；`gpus=[0,2]` 表示每张 GPU 同时运行一个实验，`[-1]` 使用 CPU。
- `DATASET_PARAMS`：四个数据集各自独立的方法参数；初始值沿用 SHOT object 分支默认值，`hidden_dim=256` 为 GCN 适配设置。

依赖 EMBER 环境及 torchvision；数据路径沿用 `EMBER/dataset.py`。命令行参数覆盖本次所有所选数据集的相应设置，例如：

```bash
python run.py --datasets pokec --runs 5 --gpus 0 --target-lr 0.001
```

每次运行新建 `results/<UTC时间戳>_<source_protocol>/`，保存：

- `summary.md`：最终 ACC/AUC/DP/EO 的 mean +/- std 主表。
- `results.json`：实际配置、逐种子指标、源选模信息及统计结果。

失败时另存对应 `.log` 并报错，不生成缺失运行的主表。

## 方法与训练协议

结果标为 **SHOT-GCN**：将图像 backbone 替换为两层 GCN-ReLU，保留 SHOT object 分支的瓶颈、分类头、label smoothing、SGD/调度、原型聚类及目标损失。

GCN 使用加载器返回的边，补齐缺失自环并按全图度数对称归一化。每个 minibatch 保留中心节点完整的两跳邻域，只缓存固定的 `S @ X`，与标准两层 GCN 对相同节点的输出及参数梯度等价（允许浮点误差）。瓶颈 BatchNorm 和信息最大化仍按中心节点 minibatch 计算。

源阶段以完整源图为上下文，仅使用所选源标签监督；目标阶段释放源样本及源图，只使用无标签目标图，冻结分类头 C、更新 F/B。目标标签、敏感属性和评价 mask 不参与适配，最终迭代用于评价。

| source_protocol | 源监督标签 | 源选模 | 对齐范围 |
|---|---|---|---|
| full（默认） | 全部有效 0/1 源标签 | val_mask 总体 ACC 最优，平分取较晚者；选模子集参与训练 | EMBER 主实验 |
| train_val | train_mask 内有效源标签 | 相同准则，val_mask 独立留出 | DANCE、FatraGNN、GraphCTA |

`full` 的选模分数不是独立验证性能；论文需说明不同方法的源监督预算差异。源选模使用总体 ACC，最终报告的 ACC 使用类别平均。

`--source-protocol train_val --source-only` 仅执行源验证，不执行目标适配或评分；该模式的结果不能作为目标域结果。

## 数据与评价

复用 `EMBER.dataset.get_dataset`，与 DANCE、FatraGNN、GraphCTA 当前入口一致。

| 数据集 | 源 → 目标 | 加载器特殊处理 |
|---|---|---|
| bailA | _2 → _1 | 移除 user_id、WHITE、RECID；标签沿用原值 |
| germanA | _2 → _1 | 标签 -1→0；Female→1、Male→0；移除 Gender、GoodCustomer、OtherLoansAtStore、PurposeOfLoan；内部划分种子 20 |
| pokec | _z → _n | 按 z 表头顺序取两域共同列，移除 user_id、region、预测属性；标签 >1→1，-1 保留；沿用 user_id 到行索引的边映射 |
| syn | -2 → -1 | 不额外二值化标签或敏感属性 |

四个数据集均逐域逐列归一化到 [-1,1]，分母 clamp_min(1e-12)，常数列为 -1；不额外插补或标准化，NaN/Inf 直接报错。Pokec 的 `y=-1` 节点仅从源监督、源选模和最终评价中排除，仍参与图传播和目标适配。共同表头所需的两域读取在训练前完成。

评价目标 `train_mask | val_mask | test_mask` 内有效 0/1 标签节点：

- ACC：类别准确率平均值；预测阈值为 `prob > 0.5`。
- AUC：使用连续 softmax 概率，按类别平均。
- DP/EO：调用 `EMBER.utils.fair_metric`，比较敏感组 0、1；其他敏感值不影响 ACC/AUC 的评价范围。
- 以百分数报告均值和总体标准差（ddof=0）；未定义指标保留 NaN（JSON 为 null），不删除相应种子。

默认源种子为 1111、2222、…，目标种子另加 100000。超参数选择应使用独立开发任务或预先规定的规则，不按目标测试结果选模。
