# Cross-Method t-SNE Visualization

这个目录放论文可视化部分的跨方法 t-SNE 代码、输入 embedding 和生成结果。

## Directory Layout

```text
visualization/
  config.yaml
  run_tsne.py
  data_io.py
  plotting.py
  export_utils.py
  embeddings/
    <method>/
      <dataset>/
        feat.npz
        labels.npz
  results/
    <method>_<dataset>/
      coordinates.csv
      tsne.png
      tsne.pdf
  README.md
```

当前只考虑四个数据集：

```text
bailA
germanA
pokec
syn
```

## Input Format

每个方法、每个数据集对应一个目录：

```text
visualization/embeddings/<method>/<dataset>/
```

里面固定放两个文件：

```text
feat.npz
labels.npz
```

`feat.npz` 默认 key：

```text
representations
```

shape：

```text
[num_nodes, embedding_dim]
```

`labels.npz` 默认 key：

```text
labels
```

shape：

```text
[num_nodes]
```

`labels` 默认编码：

```text
0 -> Y=1, S=0
1 -> Y=1, S=1
2 -> Y=0, S=0
3 -> Y=0, S=1
```

如果某个方法更方便直接导出 `y` 和 `sens`，也可以把 `y` 和 `sens` 放在 `labels.npz` 或 `feat.npz` 中；读取器会自动编码为上述四类。

支持的默认 key：

```text
y / labels_y / target / targets
sens / s / sensitive / sens_labels
```

## Output Format

每个 `{method}_{dataset}` 单独一个结果目录：

```text
visualization/results/<method>_<dataset>/
```

生成文件：

```text
coordinates.csv
tsne.png
tsne.pdf
```

例如：

```text
visualization/results/SFFGNN_pokec/coordinates.csv
visualization/results/SFFGNN_pokec/tsne.png
visualization/results/SFFGNN_pokec/tsne.pdf
```

图形样式与 `../../paper/tsne/CELL_pokec.pdf` 一致：单张 t-SNE 散点图、无标题、无图例、坐标轴归一化到 0.0-1.0；使用 Times New Roman，刻度为 18 pt，底部标签为 20 pt，散点大小为 20、边宽为 0.3，坐标轴边框为 1 pt 灰色。四类颜色依次为 coral、cornflowerblue、darkseagreen 和 pink。底部标签格式为：

```text
<method> <dataset title> t-SNE result
```

其中 `<dataset title>` 读取 `config.yaml` 中对应数据集的 `title` 字段；输入路径和结果目录仍使用 `name` 字段。

## Usage

检查依赖：

```bash
python visualization/run_tsne.py --check-deps
```

列出配置中的方法和数据集：

```bash
python visualization/run_tsne.py --list
```

使用 `config.yaml` 中 `enabled: true` 的方法和数据集：

```bash
python visualization/run_tsne.py
```

指定方法和数据集：

```bash
python visualization/run_tsne.py --datasets pokec --methods SFFGNN,DGSDA,HGDA
```

同时生成多个数据集：

```bash
python visualization/run_tsne.py --datasets bailA,germanA,pokec,syn --methods SFFGNN
```

调整采样点数：

```bash
python visualization/run_tsne.py --datasets pokec --methods SFFGNN --max-points 3000
```

缺文件时直接失败：

```bash
python visualization/run_tsne.py --datasets pokec --methods SFFGNN,DGSDA --strict
```

## Export Helper

SFFGNN 或 baseline 可以复用 `export_utils.py`：

```python
from visualization.export_utils import save_visualization_embeddings

all_mask = (
    data.train_mask | data.val_mask | data.test_mask
) & (data.y >= 0)

save_visualization_embeddings(
    "visualization/embeddings",
    method="SFFGNN",
    dataset=args.dataset,
    representations=feat[all_mask].cpu().numpy(),
    y=data.y[all_mask].cpu().numpy(),
    sens=data.sens_labels[all_mask].cpu().numpy(),
)
```

target 节点范围统一为 train、validation 和 test 的并集，并排除 Pokec 中标签为 `-1` 的节点。模型应先在完整 target 图上计算表示，再使用同一个 `all_mask` 截取 feature、`y` 和 `sens`。

这会写出：

```text
visualization/embeddings/SFFGNN/<dataset>/feat.npz
visualization/embeddings/SFFGNN/<dataset>/labels.npz
```

当前约定不区分 `source_trained` 和 `adapted`。如果要画哪一种表示，请手动保证 `feat.npz` 和 `labels.npz` 中放的是对应阶段的数据。

## Adding A Baseline

新增 baseline 时，只需要准备标准 embedding 目录：

```text
visualization/embeddings/<NewMethod>/<dataset>/feat.npz
visualization/embeddings/<NewMethod>/<dataset>/labels.npz
```

然后在 `config.yaml` 中加入：

```yaml
  - name: NewMethod
    enabled: true
```

如果文件 key 不叫 `representations` 和 `labels`，可以为该方法单独配置：

```yaml
  - name: NewMethod
    enabled: true
    embedding_key: z
    label_key: group
```

如果路径不遵守标准目录，也可以覆盖路径：

```yaml
  - name: NewMethod
    enabled: true
    embedding_paths:
      - "{project_root}/somewhere/{method}/{dataset}/embedding.npz"
    label_paths:
      - "{project_root}/somewhere/{method}/{dataset}/label.npz"
```

可用占位符：

```text
{project_root}
{visualization_root}
{embeddings_root}
{results_root}
{method}
{dataset}
```
