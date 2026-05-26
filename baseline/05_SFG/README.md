# SFG
A PyTorch implementation of "[Stable Fair Graph Representation Learning with Lipschitz Constraint](https://openreview.net/pdf?id=oJQWvsStNh)"

## Overview
SFG is a Lipschitz constraint-based method to maintain the stability of fair GNNs while preserving fgairness and accuracy performance **The core idea behind SFG is to control the size of the encoder weights space in the presence of a generator**.

## Datasets
We have provided a *rar* formatted compressed file, which you can simply extract directly.

## Reproduction
To reproduce our results, please run:
```shell
bash run.sh
```

## Visualize
If you want to visualize the result comparison, you can do it as follows:

1. Copy the log to the visualize folder:   
Navigate to the *logs* directory and locate the training log file named *XXXXX_ResLog.txt*. Then, Copy the corresponding log to the *visualize* folder.

2. Run the Plot Script: run *plot_curve_30.ipynb*.                               
For simplicity, we provide some sample reference data that you can use directly.

Note: It is important that the new file names are consistent to ensure the script reads the correct data.
