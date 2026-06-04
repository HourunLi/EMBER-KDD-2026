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



## UDAGCN

Adversarial UDA
Feature-level Alignment
适配：GCN + PPMI + Entropy; 删去L_DA
or 加入部分SFDA方法



## GRADE

Discrepancy-based Graph Domain Adaptation