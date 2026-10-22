# SHOT 的 EMBER 数据接口

在 `run.py` 顶部修改 `DEFAULTS`，然后在本目录运行：

```bash
python run.py
```

最常修改的参数是 `datasets`、`runs`、`gpus`、`source_epochs`、
`target_epochs`、`batch_size` 和 `output_dir`。默认运行全部四个数据集，
每个数据集 5 次。`gpus = [0, 2]` 表示每张 GPU 同时执行一个独立运行，
`gpus = [-1]` 表示在 CPU 上依次执行。GPU 编号相对于当前进程的
`CUDA_VISIBLE_DEVICES`；脚本不重写该环境变量。

也支持命令行覆盖，例如：

```bash
python run.py --datasets bailA germanA --runs 5 --gpus 0 2
python run.py --datasets syn --runs 1 --gpus -1 --source-epochs 2 --target-epochs 2
```

需要可运行 EMBER 的 Python 环境，以及原始 SHOT `object/network.py`
所导入的 `torchvision`。启动器使用同一个 Python 解释器创建子进程。
数据路径直接沿用 `EMBER/dataset.py`，默认在该文件配置的
`/home/user/zhangyutong/dataset/` 下读取。运行器不改变路径或预处理代码。

## 数据和评估

| 数据集 | 源域 → 目标域 | 保留的 EMBER 特殊处理 |
|---|---|---|
| bailA | `bailA_2` → `bailA_1` | 移除 `user_id`、`WHITE`、`RECID` 特征列；监督标签沿用原值 |
| germanA | `germanA_2` → `germanA_1` | 标签 `-1 → 0`；`Female → 1`、`Male → 0`；移除 `Gender`、`GoodCustomer`、`OtherLoansAtStore`、`PurposeOfLoan`，其余列与原加载器一致；保留加载器内部固定的划分种子 20 |
| pokec | `pokec_z` → `pokec_n` | 两域共有特征列；移除 `user_id`、`region`、预测属性；标签 `>1 → 1`，保留 `-1` 缺失标签；`-1` 节点不参与源域监督和最终评估，但其目标域特征参与无监督适配 |
| syn | `syn-2` → `syn-1` | 直接读取原特征、标签、敏感属性；不额外二值化标签或敏感属性 |

四个数据集都只调用一次原始 `get_dataset` 的 `feature_norm`：每个域分别
按列缩放到 `[-1, 1]`，保留 `clamp_min(1e-12)` 的常数列处理。
加载器的图边、自环、特征顺序和按类别 80%/10%/10% 划分全部保留。
SHOT 是基于样本特征的方法，加载后的图边不参与网络前向。

源域使用加载器的 `train_mask` 训练，按 SHOT 原有的源域验证集整体准确率
选择源域参数。源域训练完成后释放源域数据。目标域阶段只接收 `target.x`，
不接收真实标签、敏感属性、划分掩码或源域样本，最终使用最后一次目标迭代
的模型评估。

最终统计范围是目标域的 `train_mask | val_mask | test_mask` 中具有有效
`{0, 1}` 标签的节点，与 EMBER 的 `all` 范围一致。ACC 按当前
`EMBER/learn.py` 对两个类别分别计算准确率再平均。AUC 使用
`softmax(logits)[:, 1]` 的连续概率并按 EMBER 进行类别平均，预测使用
`prob > 0.5`。DP 和 EO 直接调用 `EMBER.utils.fair_metric`；其中 EO 是
标签 0、标签 1 下两个群体正确预测率差值的平均。非 0/1 敏感值不强制转换，
公平性函数只比较敏感组 0 和 1。四项指标均以百分数输出。

种子与当前 EMBER 的规则一致：源域 `1111 + run_index * 1111`，
目标域再加 `100000`，每次运行分别设置数据加载和训练种子。
汇总使用总体方差和总体标准差（`ddof=0`）。若必要群体为空，未定义指标
保留为 NaN，JSON 中表示为 `null`；不把它填为 0，也不静默丢弃该次运行。

## SHOT 保留的部分和输入适配

原仓库只提供图像 CNN backbone。为接收节点向量，`run.py` 的 `netF`
使用 `Linear → ReLU → Linear → ReLU` 的 MLP。原始 SHOT 文件不作修改。
这是对图像输入网络的必要适配，运行结果对应使用此 MLP backbone 的 SHOT。

`netB`、`netC`、权重初始化、源域 label smoothing、SGD 和多项式学习率
调度直接复用 `object/` 下的实现。目标阶段保留冻结 `netC`、分别设置
`netF/netB` 学习率、信息最大化损失、按间隔刷新伪标签，以及原始两轮
原型聚类。聚类按预测数量筛选类别，余弦距离保留拼接常数列和 L2 归一化。
只移除了原代码使用目标域真值打印准确率的诊断，不使用目标域真值选择参数。
`cls_par = 0` 对应原方法的 SHOT-IM 分支。

## 输出

`output_dir` 默认为本目录下的 `results/`：

- `config.json`：本次完整运行参数。
- `<dataset>/run_<n>/run.log`：每次运行的训练日志和最终指标。
- `<dataset>/run_<n>/metrics.json`：该次运行的种子、源域选择信息和四项指标。
- `<dataset>/run_<n>/source.pt`、`target.pt`：模型参数，可通过
  `save_checkpoints = False` 关闭。
- `summary.md`：本次运行的各数据集均值 ± 标准差和逐次结果。
- `summary.json`：均值、方差、标准差、逐次结果和运行参数。

每次启动都会重新训练；汇总只读取本次成功完成的运行，不把旧运行混入统计。
需要保留不同实验时，为每次实验设置不同的 `output_dir`。
