# baseline

## GraphAny
domain generalization


GraphAny SFDA Benchmark Results (3 runs, mean±std, final-epoch checkpoint)
|Dataset|Src→Tgt|ACC|AUC|DP|EO|
|-|-|-|-|-|-|
|pokec|pokec_z -> pokec_n|69.58±1.15%|75.80±1.28%|4.71±1.48%|2.72±1.13%|
|bailA|bailA_2 -> bailA_1|85.31±0.71%|92.93±1.03%|4.03±2.61%|5.54±3.03%|
|german|german_2 -> german_1|69.23±0.00%|57.24±0.28%|7.14±0.00%|9.52±0.00%|
|syn-2|syn-2 -> syn-1|87.82±0.71%|94.54±0.51%|14.37±2.25%|14.80±4.82%|


## DGSDA
UGDA（目标域可能包含源域未见过的新类别）
Attribute Alignment
Spectral Model Alignment

## HGDA

## UDAGCN
Adversarial UDA
Feature-level Alignment
适配：GCN + PPMI + Entropy; 删去L_DA
or 加入部分SFDA方法

## GRADE
Discrepancy-based Graph Domain Adaptation