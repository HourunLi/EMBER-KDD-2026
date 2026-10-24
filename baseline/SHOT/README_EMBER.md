# SHOT-MLP：EMBER 四数据集复现

修改 `run.py` 顶部的配置，在本目录执行 `python run.py`：

- `DEFAULTS`：共用的运行设置，如数据集列表、运行次数、GPU、种子和源监督协议。
- `DATASET_PARAMS`：`bailA`、`germanA`、`pokec`、`syn` 各自独立的完整方法参数。
  只修改某个数据集的字典，即只影响该数据集。

四份方法参数均初始化为 SHOT object 分支原始默认值：源/目标 lr=0.01、
源 F 倍率=0.1、目标 F/B 倍率=0.1/1.0、源/目标 epoch=20/15、
batch_size=64、bottleneck=256、smooth=0.1、cls_par=0.3、ent_par=1.0、
ent/gent=True、epsilon=1e-5、threshold=0、distance=cosine、classifier=bn、
layer=wn、interval=15。MLP 的 hidden_dim=256 是当前向量输入适配设置，
原 CNN 没有这个参数。种子、源监督协议及计算资源设置继续沿用当前复现约定。

默认四个数据集各 5 次；`gpus=[0,2]` 表示每张 GPU 同时运行一个实验，
`gpus=[-1]` 为 CPU。需要 EMBER 环境及 torchvision，数据路径沿用
`EMBER/dataset.py`。命令行也可覆盖配置：

```bash
python run.py --datasets bailA germanA pokec syn --runs 5 --gpus 0 2
```

显式传入的方法参数会覆盖本次所有所选数据集的对应值。例如
`--datasets pokec --target-lr 0.001` 只覆盖本次 Pokec 的目标学习率；
未传入的方法参数使用其 `DATASET_PARAMS` 中的值。

每次启动在 `results/<UTC时间戳>_<source_protocol>/` 中只生成：

- `summary.md`：最终目标模型的 ACC/AUC/DP/EO，mean +/- std 主表。
- `results.json`：一次保存共用配置、`config.dataset_params` 中各数据集实际使用的
  超参数、逐种子最终指标、源选模信息和统计结果。

不再输出适配前指标、混淆矩阵、模型权重、预测数组或成功运行的训练日志。
只有失败时保存对应 `.log`，失败运行不会被跳过后生成不完整主表。
历史结果不会覆盖。源验证模式会在主表中明确标注为源验证，不能充当目标结果。

## 方法及评价协议

结果标为 **SHOT-MLP**。节点向量使用两层 Linear-ReLU MLP，保留 SHOT
object 分支的瓶颈、分类头、label smoothing、SGD/调度、两轮原型聚类和目标损失。
它没有图消息传播，也不等同于原图像实验的预训练 CNN。
目标阶段冻结分类头，仅更新 F/B；保留跳过 singleton batch、伪标签刷新间隔和
原周期推理，使用最终迭代。仅输出结果不会改变训练中的原有推理和随机采样顺序。

| source_protocol | 源类别监督 | 源选模 | 对齐范围 |
|---|---|---|---|
| full（默认） | 全部有效 0/1 源标签 | val_mask 总体 ACC 最优，平分取较晚者；选模子集参与训练 | EMBER 主实验 |
| train_val | train_mask 内有效源标签 | 同样准则，val_mask 独立留出 | DANCE、FatraGNN、GraphCTA |

full 是 SHOT 原生支持的训练约定，其选模分数不是独立验证性能。
仅修改 SHOT 无法消除主实验与三个 baseline 已有的源监督预算差异，论文需说明。
`source_selection.selection_ACC` 是原 SHOT 总体准确率选模分数，
论文 `metrics.ACC` 则是类别准确率平均值。

两域在学习前分别设种子并通过原 EMBER 加载器准备。源训练后清除源梯度及样本，
目标优化函数只接收模型和全部目标特征，不接收目标标签、敏感属性或评价 mask，
也不再访问任一域的原始文件。目标真值仅用于适配结束后的评价。

评价范围为目标 `train_mask | val_mask | test_mask` 内有效 0/1 标签节点。
ACC 是类别准确率平均值；AUC 使用连续 softmax 概率并按类别平均；
预测严格为 `prob > 0.5`；DP/EO 直接调用 `EMBER.utils.fair_metric`。
非 0/1 敏感值保留，公平指标比较组 0、1，ACC/AUC 不因此删除节点。
百分数输出，方差/标准差使用 ddof=0；未定义指标为 NaN（JSON 为 null），
不填 0 或删除该种子。源种子默认 1111、2222、…，目标种子再加 100000。

## 数据核查，特别是 Pokec

实际入口与 DANCE/sf.py、FatraGNN/sf.py、GraphCTA/run.py 一样复用
`EMBER.dataset.get_dataset`，不采用这些方法旧的独立数据加载入口。

| 数据集 | 源 → 目标 | 原加载器特殊处理 |
|---|---|---|
| bailA | _2 → _1 | 移除 user_id、WHITE、RECID；标签沿用原值 |
| germanA | _2 → _1 | 标签 -1→0；Female→1、Male→0；移除 Gender、GoodCustomer、OtherLoansAtStore、PurposeOfLoan；固定内部划分种子 20 |
| pokec | _z → _n | 按 z 表头顺序取两域共同列，移除 user_id、region、预测属性；标签 >1→1，-1 保留 |
| syn | -2 → -1 | 不额外二值化标签或敏感属性 |

四个数据集均逐域逐列归一化到 [-1,1]，分母 clamp_min(1e-12)，常数列为 -1。
不重新标准化、不额外插补特征缺失编码；若加载后出现 NaN/Inf，运行器直接报错。
Pokec 外部 user_id 到行索引的图映射也沿用原加载器，但 SHOT 不使用图边。

Pokec 的 `y=-1` 不应改成 0 或第三个预测类别：源监督/源选模及最终评价排除它，
目标适配仍保留它的特征。不能为改善结果按目标真值筛出“有标签节点”再适配。
共同表头所需的两域读取在离线准备完成，符合随后目标阶段只使用目标特征的边界。
本地没有真实 CSV，核查覆盖实现和通过实际加载器的构造样例，不代表已验证服务器原始文件。

## Pokec 参数对照的规范边界

本次不更改任何训练数值默认值：target_lr=0.01、target_epochs=15、batch_size=64、
cls_par=0.3、ent_par=1；也不加入 Pokec 自动特判、目标分数早停或 source-only 回退。

已有旧版 Pokec 日志显示每个目标 epoch 有 1041 个 batch，共 15615 次更新，
无标签预测熵逐渐下降。它不是最近一次 full 协议的逐次日志，不能据此确定当前退化机制。
缺失标签较多时，全部节点适配与有效标签节点评价的样本组成可能不同；
伪标签反馈、信息最大化及 BatchNorm 更新可能放大错误，不应通过更改标签处理来规避。

若开展补充实验，首先仅比较一个有官方配置依据的目标学习率：

```bash
python run.py --datasets pokec --runs 5 --gpus 0 --target-lr 0.001
```

依据是 `object/run.sh` 的大规模 VisDA 配置使用 1e-3；这仅为候选值依据，
不是已验证适合 Pokec 的最优参数，也没有照搬 VisDA 的网络和源训练配置。
先固定这一对照、其余参数及全部种子，再运行并保留与 0.01 的完整比较。
已看到目标分数后开展的实验应披露为补充/敏感性分析，不能反复挑选最高分替换默认主结果。
仅凭无标签熵较低也不能确认配置较优。

其次可研究 batch_size=128/256 或较短的预设目标训练预算；当前 batch_size 同时影响
源/目标两阶段，不能把它的变化全部归因于目标适配。修改 target_epochs 还会改变调度和
伪标签刷新间隔，不等价于从原日志里挑较早 checkpoint。不要同时修改多个因素后归因。
目标阶段参数若要作为正式最终设置，应使用独立开发任务或事先规定的规则选择。
源学习率、源训练预算等则可用独立源验证选择：

```bash
python run.py --datasets pokec --source-protocol train_val --source-only --gpus 0
```

此模式不加载目标域入口、不进行目标推理或评分；Pokec 源加载器仍可能读取两域表头。
源验证不能用于选择仅在目标阶段生效的学习率、损失权重或目标 epoch。
