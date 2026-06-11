# baseline

## GraphAny

domain generalization

移除 target 阶段对 target labels 的依赖，直接使用

GraphAny SFDA Benchmark Results (3 runs, mean±std, final-epoch checkpoint)
|Dataset|ACC|AUC|DP|EO|
|-|-|-|-|-|
|bailA|77.51±0.18|88.23±0.98|1.58±0.09|1.05±0.31|
|german|69.78±0.12|64.85±0.20|19.96±0.84|17.64±0.51|
|pokec|67.31±0.17|74.68±0.11|1.44±0.32|1.79±0.43|
|syn|86.79±0.25|94.89±0.09|18.44±0.39|20.97±0.38|



## DGSDA

UGDA（目标域可能包含源域未见过的新类别）

Attribute Alignment & Topology to Model Alignment

L_source(源域标签学习) & L_mmd(attribute align) & L_align(model align) & L_target(目标域聚类)

删除L_mmd

Phase 1: Source Pretraining - L_pretrain = L_source
- 属性编码器 h_X 的全部参数
- 源域 BernNet 的 Bernstein 多项式系数 {θ^S_k}_{k=0}^K
- 分类器的全部参数

Phase 2: Target Adaptation - L_SFDA = α * L_align_SF + γ * L_target

冻结参数（不参与梯度更新）:
- 属性编码器 h_X 的参数
- 源域 Bernstein 系数 {θ^S_k}（作为固定锚点）
- 分类器参数

仅优化目标域 BernNet 的 Bernstein 系数 {θ^T_k}

其中 L_align_SF 的计算方式与原始 L_align 完全相同，但 θ^S_k 使用预存的固定值而非实时从源域计算:
L_align_SF = Σ_k ||θ^S_k(fixed) - θ^T_k|| + Σ_k (||θ^S_k(fixed)|| + ||θ^T_k||)

bailA 出现 Acc约50% 而 AUC约90% 的异常现象，将lr调整为0.001

|Dataset|ACC|AUC|DP|EO|
|---|---|---|---|---|
|bailA|79.45±1.30|90.96±0.36|2.72±0.56|5.00±1.35|
|german|72.21±1.18|61.11±0.42|15.78±0.98|14.99±1.14|
|pokec|67.99±0.25|74.87±0.20|0.99±0.42|0.65±0.40|
|syn|79.39±0.17|87.83±0.13|6.32±0.17|6.55±0.42|



## HGDA

UGDA, 基于KL散度的 node-level alignment, 显式建模图同质性/异质性分布偏移

阶段一：源域预训练 + 知识蒸馏存储

在源图 $G^S$ 上用完整 HGDA 的特征提取器训练分类器，只使用源域损失：
$\mathcal{L}_{pretrain} = \mathcal{L}_S = -\frac{1}{N_S}\sum_{i=1}^{N_S} y^S_i \log \hat{y}^S_i​$

保存三个滤波器的权重 $\{W_L, W_F, W_H\}$ 及分类头 $f^S$ ，额外存储类条件原型(Class-Conditional Prototypes):

$\boldsymbol{\mu}^S_{L,c} = \frac{1}{|V^S_c|}\sum_{i \in V^S_c} Z^S_{L,i}, \quad c = 1, \ldots, C$

对三个滤波器分别计算，共存储 3C 个原型向量：

$\mathcal{P} = \left\{ \boldsymbol{\mu}^S_{L,c},\ \boldsymbol{\mu}^S_{H,c},\ \boldsymbol{\mu}^S_{F,c} \right\}_{c=1}^C​$

阶段二：目标域无源自适应

目标图 $G^T$ 可用，加载预训练权重 $\{W_L, W_F, W_H, f^S\}$ 及存储的 $\mathcal{P}$ ​。

改造说明:

1. 替换损失：基于原型的同质性对齐损失 $\mathcal{L}^{SF}_H$
   用伪标签将目标节点分配到各类，计算目标嵌入与存储原型的 KL 散度，替代原来对源域在线嵌入的依赖：

   $\mathcal{L}^{SF}_H = \sum_{c=1}^C \hat{p}(c) \cdot \left[ KL\!\left(\hat{Z}^T_{L,c} \,\Big\|\, \mathcal{N}(\boldsymbol{\mu}^S_{L,c}, \mathbf{I})\right) + KL\!\left(\hat{Z}^T_{H,c} \,\Big\|\, \mathcal{N}(\boldsymbol{\mu}^S_{H,c}, \mathbf{I})\right) + KL\!\left(\hat{Z}^T_{F,c} \,\Big\|\, \mathcal{N}(\boldsymbol{\mu}^S_{F,c}, \mathbf{I})\right) \right]$

   其中 $\hat{Z}^T_{L,c}$​ 表示被伪标签分配到类 c 的目标节点嵌入的均值，$\hat{p}(c)$ 是目标域的估计类别先验。

2. 保留损失：目标域熵最小化 $\mathcal{L}_T$
   
   $\mathcal{L}_T = -\frac{1}{N_T}\sum_{i=1}^{N_T} \hat{y}^T_i \log \hat{y}^T_i$

阶段二总损失:
$\mathcal{L}_{hgdasf} = \mathcal{L}^{SF}_H + \alpha \mathcal{L}_T$

注：阶段二冻结 classifier; lr 拆分为 source_lr 和 target_lr, 并将 target_lr 设为 0.001

|Dataset|ACC|AUC|DP|EO|
|---|---|---|---|---|
|bailA|88.35±1.80|93.05±2.79|4.00±0.34|6.35±0.61|
|german|62.68±1.30|57.46±1.34|14.16±4.07|14.57±5.26|
|pokec|65.91±0.89|70.57±0.32|1.79±0.80|1.59±0.48|
|syn|92.85±0.04|96.72±0.07|4.81±0.13|4.30±0.65|



## UDAGCN

adversarial feature-level alignment

贡献一：局部+全局双通道 GCN + 图间注意力（解决图表示学习层面的问题）\
贡献二：三损失对抗域自适应框架（解决域迁移层面的问题）

删去对抗对齐损失L_DA

ppmi_conv 修复了 german 报 NaN 的问题

阶段二冻结分类器 + source-prior marginal regularization (lambda_prior = 0.4), 避免预测坍塌为一类

|Dataset|ACC|AUC|DP|EO|
|---|---:|---:|---:|---:|
|bailA|87.62±0.97|87.46±1.32|4.17±1.08|5.49±1.47|
|german|63.55±2.18|52.25±2.76|20.19±5.14|20.47±4.70|
|pokec|54.70±0.43|55.96±0.58|14.10±0.22|9.50±1.28|
|syn|86.07±1.15|92.01±1.13|24.17±2.71|15.80±3.85|



## GRADE

non-IID UDA

基于 Weisfeiler-Lehman (WL) 子树核与消息传递 GNN 的图子树差异 Graph Subtree Discrepancy (GSD)