# baseline

## GraphAny

domain generalization

直接使用

GraphAny SFDA Benchmark Results (3 runs, mean±std, final-epoch checkpoint)
|Dataset|ACC|AUC|DP|EO|
|-|-|-|-|-|
|BailA|85.31±0.71%|92.93±1.03%|4.03±2.61%|5.54±3.03%|
|German|69.23±0.00%|57.24±0.28%|7.14±0.00%|9.52±0.00%|
|Pokec|69.58±1.15%|75.80±1.28%|4.71±1.48%|2.72±1.13%|
|syn|87.82±0.71%|94.54±0.51%|14.37±2.25%|14.80±4.82%|



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



## HGDA



## UDAGCN

Adversarial UDA
Feature-level Alignment
适配：GCN + PPMI + Entropy; 删去L_DA
or 加入部分SFDA方法



## GRADE

Discrepancy-based Graph Domain Adaptation