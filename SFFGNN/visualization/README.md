# SFFGNN t-SNE 可视化工具

这个目录实现一套独立的 t-SNE 可视化流程，用来复现 CELL 论文 4.4 Visualization 的处理方式：在目标域图上读取各方法的 source-trained node embeddings，用 t-SNE 降到二维，并按 `(Y, S)` 四个任务标签/敏感属性组合着色，观察表示空间中不同群体是否被明显分隔。

CELL 4.4 的核心设定是：对目标域 Pokec-n 的 source-trained representations 做 t-SNE，可视化 Fatra、DANCE、CELL 三个方法，点颜色对应 `(Y, S)` group。本文工具沿用同一逻辑，但方法和数据集都由 `config.yaml` 或命令行选择。

## 文件说明

- `run_tsne.py`：命令入口，读取配置、加载 embedding、运行 t-SNE、输出图。
- `config.yaml`：数据集、方法、路径模板、t-SNE 参数、采样和绘图参数。
- `data_io.py`：统一读取 `.npz` embedding/label 文件，并做采样和 CSV 坐标输出。
- `plotting.py`：t-SNE 调用和多方法并排散点图。
- `export_utils.py`：给 SFFGNN 和 baseline 训练脚本复用的标准导出函数。

输出默认写入：

```text
SFFGNN/results/visualization/
```

其中图片文件形如：

```text
SFFGNN/results/visualization/pokec_source_trained_tsne.png
SFFGNN/results/visualization/pokec_source_trained_tsne.pdf
```

每个方法的二维坐标也会保存为 CSV：

```text
SFFGNN/results/visualization/coordinates/pokec/SFFGNN_source_trained_tsne.csv
```

## 依赖

运行可视化需要：

```bash
pip install numpy matplotlib scikit-learn pyyaml
```

可以先检查依赖：

```bash
python SFFGNN/visualization/run_tsne.py --check-deps
```

## 基本用法

列出配置中的数据集和方法：

```bash
python SFFGNN/visualization/run_tsne.py --list
```

使用配置里 `enabled: true` 的数据集和方法绘图：

```bash
python SFFGNN/visualization/run_tsne.py
```

指定目标数据集和方法：

```bash
python SFFGNN/visualization/run_tsne.py --datasets pokec --methods SFFGNN,DGSDA,HGDA
```

指定 stage，例如比较 adapted embeddings：

```bash
python SFFGNN/visualization/run_tsne.py --datasets pokec --methods SFFGNN,DGSDA --stage adapted
```

大图上 t-SNE 很慢，默认每个方法最多分层采样 5000 个点。可以调整：

```bash
python SFFGNN/visualization/run_tsne.py --datasets pokec --methods SFFGNN --max-points 3000
```

如果希望某个方法缺少文件时直接失败：

```bash
python SFFGNN/visualization/run_tsne.py --datasets pokec --methods SFFGNN,DGSDA --strict
```

## 标准输入格式

每个方法需要给每个数据集导出两个 `.npz` 文件：

```text
<method_root>/visualization_embeddings/<dataset>_<stage>_feat.npz
<method_root>/visualization_embeddings/<dataset>_<stage>_labels.npz
```

`feat.npz` 必须包含：

```text
representations: shape = [num_nodes, embedding_dim]
```

`labels.npz` 必须包含：

```text
labels: shape = [num_nodes]
```

`labels` 的编码默认采用 SFFGNN 现有导出逻辑：

```text
0 -> Y=1, S=0
1 -> Y=1, S=1
2 -> Y=0, S=0
3 -> Y=0, S=1
```

如果你的 baseline 更容易导出原始 `y` 和 `sens`，也可以在一个 `.npz` 中保存 `y` 和 `sens` 两个数组；加载器会自动编码为上述四类。支持的默认 key 包括：

```text
y / labels_y / target / targets
sens / s / sensitive / sens_labels
```

## SFFGNN 导出

SFFGNN 的 `runner.py` 已经接入标准导出，但默认不写 `visualization_embeddings`，避免 `main.py` 和 `tune.py` 在常规训练、调参时产生额外文件。需要导出时显式打开：

```bash
python SFFGNN/main.py --save_visualization_embeddings
```

开关打开后，目标域适配前评估会写：

```text
SFFGNN/visualization_embeddings/<dataset>_source_trained_feat.npz
SFFGNN/visualization_embeddings/<dataset>_source_trained_labels.npz
```

目标域适配后评估会写：

```text
SFFGNN/visualization_embeddings/<dataset>_adapted_feat.npz
SFFGNN/visualization_embeddings/<dataset>_adapted_labels.npz
```

旧的兼容文件仍会按原逻辑保留：

```text
{dataset}_feat.npz
{dataset}_labels.npz
```

如果要在其他 SFFGNN 评估入口或 baseline 中手动导出，可以复用：

```python
from SFFGNN.visualization.export_utils import save_visualization_embeddings

save_visualization_embeddings(
    "SFFGNN/visualization_embeddings",
    dataset=args.dataset,
    representations=feat[data.test_mask].cpu().numpy(),
    y=data.y[data.test_mask].cpu().numpy(),
    sens=data.sens_labels[data.test_mask].cpu().numpy(),
    stage="source_trained",
)
```

## 新增 baseline 方法

如果要新增一个方法，推荐按下面的要求放置：

```text
baseline/<NewMethod>/
  ... 方法原始代码 ...
  visualization_embeddings/
    pokec_source_trained_feat.npz
    pokec_source_trained_labels.npz
    bail_source_trained_feat.npz
    bail_source_trained_labels.npz
```

这个文件夹需要满足三点：

1. `baseline/<NewMethod>/` 的文件夹名应与配置中的 `name` 一致，例如 `name: NewMethod`。
2. `visualization_embeddings/` 下的文件名遵循 `<dataset>_<stage>_feat.npz` 和 `<dataset>_<stage>_labels.npz`。
3. `feat.npz` 用 key `representations`，`labels.npz` 用 key `labels`；如果使用其他 key，需要在 `config.yaml` 中为该方法设置 `embedding_key` 和 `label_key`。

然后在 `config.yaml` 的 `methods` 中加入：

```yaml
  - name: NewMethod
    enabled: true
    root: "{baseline_root}/NewMethod"
```

如果该方法的导出路径不同，可以覆盖路径模板：

```yaml
  - name: NewMethod
    enabled: true
    root: "{baseline_root}/NewMethod"
    embedding_paths:
      - "{method_root}/runs/{dataset}/{stage}/embedding.npz"
    label_paths:
      - "{method_root}/runs/{dataset}/{stage}/labels.npz"
    embedding_key: z
    label_key: group
```

配置中可用的占位符包括：

```text
{project_root}   项目根目录 AAAI-2026
{sffgnn_root}    SFFGNN 目录
{baseline_root}  baseline 目录
{method_root}    当前方法目录
{method}         当前方法名
{dataset}        当前数据集名
{stage}          当前阶段，例如 source_trained/adapted
```

## 数据集配置

在 `config.yaml` 的 `datasets` 中控制数据集：

```yaml
datasets:
  - name: pokec
    title: Pokec-n
    enabled: true
  - name: bail
    title: Bail
    enabled: false
```

也可以不改配置，直接在命令行指定：

```bash
python SFFGNN/visualization/run_tsne.py --datasets pokec,bail --methods SFFGNN,DGSDA
```

