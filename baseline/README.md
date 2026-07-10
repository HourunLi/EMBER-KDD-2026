# baseline

## GraphAny

domain generalization

移除 target 阶段对 target labels 的依赖，直接使用

GraphAny SFDA Benchmark Results (5 runs, mean±std, final-epoch checkpoint)
|Dataset|ACC|AUC|DP|EO|
|---|---|---|---|---|
|bailA|77.79±0.21|90.79±0.99|1.68±0.21|1.51±0.69|
|germanA|70.16±0.21|61.86±0.06|5.47±0.49|4.97±0.53|
|pokec|67.53±0.45|74.70±0.10|1.16±0.28|1.72±0.37|
|syn|87.87±0.31|95.44±0.09|15.78±0.43|17.33±0.97|



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
|bailA|80.01±1.23|91.10±0.30|2.83±0.41|5.18±0.92|
|germanA|72.03±0.94|61.64±0.37|14.84±1.50|13.26±2.01|
|pokec|68.05±0.18|74.92±0.23|0.87±0.47|0.73±0.32|
|syn|79.61±0.31|87.94±0.17|5.82±0.89|6.17±0.87|



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
   使用目标预测的 hard pseudo-label 将目标节点分配到各类，计算每类目标嵌入均值与源域存储原型之间的 KL 散度，替代原来对源域在线嵌入的依赖。代码中不是对高斯分布 $\mathcal{N}(\mu, I)$ 做 KL，而是在隐藏维度上先做 softmax，再计算目标均值向量与源域原型向量的 KL：

   $\mathcal{L}^{SF}_H = \sum_{c=1}^C \hat{p}(c) \cdot \left[ KL\!\left(\mathrm{softmax}(\boldsymbol{\mu}^T_{L,c}) \,\Big\|\, \mathrm{softmax}(\boldsymbol{\mu}^S_{L,c})\right) + KL\!\left(\mathrm{softmax}(\boldsymbol{\mu}^T_{H,c}) \,\Big\|\, \mathrm{softmax}(\boldsymbol{\mu}^S_{H,c})\right) + KL\!\left(\mathrm{softmax}(\boldsymbol{\mu}^T_{F,c}) \,\Big\|\, \mathrm{softmax}(\boldsymbol{\mu}^S_{F,c})\right) \right]$

   其中 $\boldsymbol{\mu}^T_{L,c}$​ 表示被 hard pseudo-label 分配到类 c 的目标节点嵌入均值，$\boldsymbol{\mu}^S_{L,c}$ 表示源域保存的类条件原型，$\hat{p}(c)$ 是目标域 hard pseudo-label 估计出的类别占比。

2. 保留损失：目标域熵最小化 $\mathcal{L}_T$
   
   $\mathcal{L}_T = -\frac{1}{N_T}\sum_{i=1}^{N_T} \hat{y}^T_i \log \hat{y}^T_i$

阶段二总损失:
$\mathcal{L}_{hgdasf} = \mathcal{L}^{SF}_H + \alpha \mathcal{L}_T$

注：阶段二冻结 classifier; lr 拆分为 source_lr 和 target_lr, 并将 target_lr 设为 0.001

|Dataset|ACC|AUC|DP|EO|
|---|---|---|---|---|
|bailA|88.27±1.56|93.31±2.73|4.06±0.28|6.54±0.52|
|germanA|63.23±1.23|58.46±1.43|19.14±3.48|16.44±4.50|
|pokec|65.98±1.04|70.61±0.43|1.88±0.78|1.66±0.70|
|syn|92.78±0.17|96.59±0.19|5.04±0.51|4.64±1.29|



## UDAGCN

adversarial feature-level alignment

贡献一：局部+全局双通道 GCN + 图间注意力（解决图表示学习层面的问题）\
贡献二：三损失对抗域自适应框架（解决域迁移层面的问题）

删去对抗对齐损失L_DA

ppmi_conv 修复了 german 报 NaN 的问题

阶段二冻结分类器 + source-prior marginal regularization (lambda_prior = 0.4), 避免预测坍塌为一类

|Dataset|ACC|AUC|DP|EO|
|---|---:|---:|---:|---:|
|bailA|87.47±0.81|86.63±1.11|3.50±0.28|4.66±0.57|
|germanA|68.39±0.48|57.09±0.94|11.25±3.44|11.52±3.59|
|pokec|57.71±2.16|59.38±2.24|10.13±2.29|8.63±0.43|
|syn|85.72±0.94|91.53±1.12|25.01±2.51|14.40±2.99|



## GRADE

non-IID UDA

feature alignment

基于 Weisfeiler-Lehman (WL) 子树核与消息传递 GNN 的图子树差异 Graph Subtree Discrepancy (GSD): \
逐层 WL子图差异 求和，差异使用 差异距离 计算（差异距离：两个探测函数分别在源图和目标图上的期望预测差异的差值的最大值）

loss: 交叉熵 + GSD(有限M层近似)

原始 GRADE 用源图和目标图的逐层 WL 子图表示在线计算 GSD；GRADE-SF 则在源阶段预先保存逐层、类别条件的表示均值和方差，目标阶段用目标预测概率构造软伪标签统计量，再让目标逐层 class-conditional statistics 对齐到保存的源域 statistics。

GSD改造：把“在线源-目标分布差异”改成“源域统计量和目标域伪标签统计量之间的差异” \
按层、类别保存源域节点表示均值和方差 \
用 source-free GSD loss 代替原始 GSD：
$$ \mathcal{L}_{GSD}^{SF}=\frac{1}{L}\sum_{l=1}^{L}\sum_{c=1}^{C}\hat{\pi}_{c}^{T}\left[D_{\mu}(\mu_{l,c}^{T}, \mu_{l,c}^{S})+\beta D_{\sigma}(\sigma_{l,c}^{2,T}, \sigma_{l,c}^{2,S})\right] $$
$$ D_{\mu}=\frac{1}{d_l}\sum_j\frac{(\mu_{l,c,j}^{T} - \mu_{l,c,j}^{S})^2}{\sigma_{l,c,j}^{2,S} + \epsilon} $$
$$ D_{\sigma}=SmoothL1\left(\log(\sigma_{l,c}^{2,T}+\epsilon),\log(\sigma_{l,c}^{2,S}+\epsilon)\right) $$

源域预训练loss - 交叉熵 \
目标域适配loss - $\mathcal{L}_{GSD}^{SF}$

source_lr: 0.001 \
target_lr: 0.0003 \
dropout: 0.3

|Dataset|ACC|AUC|DP|EO|
|---|---:|---:|---:|---:|
|bailA|88.82+/-3.13|92.94+/-2.04|4.92+/-0.35|6.35+/-0.95|
|germanA|70.78+/-1.52|57.67+/-0.84|11.95+/-3.31|12.15+/-3.84|
|pokec|69.71+/-0.31|75.89+/-0.20|4.92+/-1.12|4.75+/-1.17|
|syn|88.16+/-0.29|95.39+/-0.03|16.00+/-0.20|13.08+/-2.09|