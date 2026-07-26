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
      <method>.png
      <method>.pdf
      candidates/
        sample_seed_<sample_seed>_tsne_seed_<tsne_seed>/
          coordinates.csv
          <method>.png
          <method>.pdf
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
<method>.png
<method>.pdf
```

例如：

```text
visualization/results/SFFGNN_pokec/coordinates.csv
visualization/results/SFFGNN_pokec/SFFGNN.png
visualization/results/SFFGNN_pokec/SFFGNN.pdf
```

图形采用与 CELL/Pilot 参考图一致的论文样式：使用 `6.4 x 4.35` 英寸画布，坐标归一化到 0.0-1.0，显示 0.0-1.0、间隔 0.2 的刻度以及完整灰色边框。刻度字号为 18 pt，底部方法/数据集标签为 20 pt；散点大小为 20 pt²。放大的四组单行图例紧贴图片框上方，并限制在图片框宽度内。输出使用 `bbox_inches="tight"` 和 0.1 英寸 padding 裁剪白边。

底部标签格式为：

```text
<method> t-SNE result on <dataset>
```

其中 `<dataset>` 使用 `config.yaml` 中的数据集 `name` 字段，与输入路径和结果目录保持一致。

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

针对固定的 embedding 尝试多个节点抽样 seed：

```bash
python visualization/run_tsne.py --datasets pokec --methods EMBER --sample-seeds 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 --tsne-seeds 0 --strict
```

传入多个 `--sample-seeds` 或 `--tsne-seeds` 时，候选结果分别写入：

```text
visualization/results/EMBER_pokec/candidates/sample_seed_<sample_seed>_tsne_seed_<tsne_seed>/
```

`sample_seed` 只控制从 embedding 中抽取哪些节点；`tsne_seed` 只控制 t-SNE 的随机状态。抽样采用分层、无放回方式。只传一个 seed 时不会创建 `candidates` 目录，而是直接生成正式的 `{method}_{dataset}` 结果。例如选定 `sample_seed=17` 后可以运行：

```bash
python visualization/run_tsne.py --datasets pokec --methods EMBER --sample-seeds 17 --tsne-seeds 0 --strict
```

论文中比较多个方法时，应对所有方法使用相同的抽样 seed，并检查各方法 `coordinates.csv` 的 `source_index` 是否一致：

```bash
python visualization/run_tsne.py --datasets pokec --methods DANCE,GraphCTA,EMBER --sample-seeds 17 --tsne-seeds 0 --strict
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

### Export From Baselines

五个 SFDA runner 均支持直接导出。以下命令运行正常的多次实验，但只有 `run_idx=0` 会写入可视化目录：

```bash
python baseline/DGSDA/dgsda_sf.py --save_visualization_embeddings --visualization_run_idx 0
python baseline/GRADE/grade_sf.py --save_visualization_embeddings --visualization_run_idx 0
python baseline/HGDA/hgda_sf.py --save_visualization_embeddings --visualization_run_idx 0
python baseline/UDAGCN/udagcn_sf.py --save_visualization_embeddings --visualization_run_idx 0
python baseline/GraphAny/run_sfda.py --save_visualization_embeddings --visualization_run_idx 0
```

各方法导出的最终 target 表示为：

```text
DGSDA    -> target Bern filter 隐藏表示
GRADE    -> 最后一层 GCN 隐藏表示（分类器之前）
HGDA     -> 三个谱分支的加权 combined 表示
UDAGCN   -> GCN/PPMI attention 融合编码
GraphAny -> attention 加权后、通道求和前的预测表示
```

输出统一写入：

```text
visualization/embeddings/<method>/<dataset>/feat.npz
visualization/embeddings/<method>/<dataset>/labels.npz
```

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
