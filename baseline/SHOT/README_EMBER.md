# SHOT-MLP 在 EMBER 四个数据集上的适配

在 `run.py` 顶部修改 `DEFAULTS`，然后进入 `baseline/SHOT` 运行：

```bash
python run.py
```

常用参数为 `datasets`、`runs`、`gpus`、`source_protocol`、训练 epoch 和
`output_dir`。默认四个数据集各运行 5 次；`gpus=[0,2]` 表示每张 GPU
同时运行一个实验，`gpus=[-1]` 为 CPU。GPU 编号相对于启动时的
`CUDA_VISIBLE_DEVICES`。也可用命令行覆盖默认值：

```bash
python run.py --datasets bailA germanA --runs 5 --gpus 0 2
python run.py --datasets syn --runs 1 --gpus -1
python run.py --source-protocol train_val --source-only --gpus 0
```

需要 EMBER 的 Python 环境及原 SHOT `network.py` 依赖的 torchvision。
数据路径直接沿用 `EMBER/dataset.py`。仅修改本目录的适配运行器和说明，
不修改 EMBER、其他 baseline 或 SHOT 的 `object/`、`digit/` 原代码。

## 方法与实验协议

结果应标为 **SHOT-MLP**：以随机初始化的两层 Linear-ReLU MLP 接收节点向量，
保留 SHOT object 分支的瓶颈、分类头、损失、优化器、学习率调度和伪标签更新。
模型不使用图边，也不等同于原图像实验中的预训练 CNN。

| source_protocol | 源域类别监督 | 源模型选择 | 对齐范围 |
|---|---|---|---|
| full（默认） | 全部有效 0/1 源标签 | 源 val_mask 总体 ACC 最优，相同分数取较晚者；选模子集参与训练 | EMBER 主实验 |
| train_val | train_mask 内有效源标签 | 相同规则，但 val_mask 是独立留出集 | DANCE、FatraGNN、GraphCTA |

`full` 对应 SHOT 原生支持的 full 训练约定，其选模分数不能当作独立验证性能。
仅修改 SHOT 无法消除主实验与三个有效 baseline 的既有源监督预算差异，论文需说明。
每次结果的 `source_selection` 记录监督范围、节点数、交集数、最佳迭代和总体
`selection_ACC`；其中 `metrics.ACC` 则按统一评估口径计算类别平均准确率。

两域预处理均在训练前完成，Pokec 的共同列名读取也在此时完成。
随后仅用源数据训练，释放源样本和梯度，再向目标适配函数传入模型和全部目标特征。
目标标签、敏感属性和评价 mask 不参与适配；Pokec 缺失标签节点的特征仍参与适配。
目标阶段冻结分类头，仅更新 F/B，保留两轮原型聚类、伪标签刷新、信息最大化、
伪标签 CE、singleton batch 跳过方式及原周期推理，最终使用最后一次目标迭代。
适配前额外推理会恢复随机数状态和模块模式；适配前后目标指标均在适配结束后计算。

## 数据与评估口径

复用未修改的 `EMBER.dataset.get_dataset` 和 `EMBER.utils.fair_metric`：

| 数据集 | 源 → 目标 | 特殊处理 |
|---|---|---|
| bailA | _2 → _1 | 移除 user_id、WHITE、RECID；标签沿用加载器 |
| germanA | _2 → _1 | 标签 -1→0；Female→1、Male→0；移除 Gender、GoodCustomer、OtherLoansAtStore、PurposeOfLoan；保留内部划分种子 20 |
| pokec | _z → _n | 共同特征列及原顺序；移除 user_id、region、预测属性；标签 >1→1，保留 -1 |
| syn | -2 → -1 | 不额外二值化标签或敏感属性 |

四个数据集均按原加载器逐域、逐列归一化至 [-1,1]，分母 clamp_min(1e-12)，
常数列为 -1；不额外插补特征。非 0/1 敏感值保留，公平指标仅比较组 0、1，
ACC/AUC 不因此删除节点。

目标评价范围是 `train_mask | val_mask | test_mask` 内有有效 0/1 标签的节点。
ACC 是类别准确率平均值，AUC 使用连续 softmax 概率按类别平均，预测阈值严格为
`prob > 0.5`；EO 是两类条件下组间正确率差的平均。指标以百分数表示，跨种子
标准差采用 `ddof=0`。未定义指标保留 NaN（JSON 为 null），不填 0 或跳过该种子。
源种子为 `1111 + run_index * 1111`，目标种子再加 100000。

## 超参数

此次精简没有改变任何训练数值默认值。源/目标学习率仍为 0.01，源骨干倍率 0.1，
目标 F/B 倍率 0.1/1，源/目标 epoch 为 20/15，batch size 64，MLP 宽度和瓶颈
均为 256，smooth=0.1，cls_par=0.3，ent_par=1。`cls_par=0` 是 SHOT-IM 独立实验。
这些是当前 object 分支参数默认值及既有 MLP 配置，不代表节点数据上的最优值。

如需选择源训练参数，可用 `train_val --source-only` 在预先固定的范围内进行源域
独立验证，固定参数后再运行目标实验；最终采用 full 时用固定配置重训。
此模式不调用目标域加载入口、不执行目标推理或评分；Pokec 源加载器仍可能读取
两域表头以确定共同特征。源骨干倍率 0.1 与 1 可作为有原实现依据的候选
（digit 分支从零训练使用相同的骨干和头部学习率），源训练预算也可通过源验证选择。
不按已知目标测试分数修改参数，不进行目标真值早停、最好种子选择或自动回退。

## 精简后的输出

每次启动在 `output_dir`（默认 `results_protocol/`）下创建独立时间戳目录，
不覆盖已下载的 `results/`：

```text
<UTC时间戳>_<source_protocol>/
  config.json             # 本次配置，仅保存一份
  summary.md              # 各数据集 mean +/- std，适配后主表和适配前诊断表
  bailA/                  # 其他数据集同样组织
    run_1.json            # 源选模、适配前后指标与混淆矩阵、目标更新预算
    run_1.log             # 源训练日志；目标 loss、entropy、预测类别数（全部目标节点）
    run_1.pt              # 可选：input_dim、source/target 两组权重，顺序均为 F/B/C
    run_1.npz             # 可选：前后 logits、labels、sensitive、eval_mask，按加载器行序
```

混淆矩阵按真实类别分行、预测类别分列，类别顺序均为 0、1。
主结果为 `target_after.metrics`，适配前为 `target_before.metrics`。
不再生成 protocol/provenance 文件、重复的配置、JSON 诊断轨迹或重复汇总 JSON。
`source_only` 只保存源选模结果，权重文件也只有源模型。
设置 `save_checkpoints=False`、`save_predictions=False`（或命令行
`--no-save-checkpoints --no-save-predictions`）即可每次只保留 JSON 和日志。
失败运行会报错并保留日志，不跳过后生成不完整的汇总。
