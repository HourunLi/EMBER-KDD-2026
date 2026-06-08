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