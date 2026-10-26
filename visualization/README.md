# Cross-Method t-SNE Visualization

This directory contains the cross-method t-SNE code, input embeddings, and generated figures used for the paper.

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
        <method>_sample_seed_<sample_seed>_tsne_seed_<tsne_seed>.png
  README.md
```

The current configuration covers four datasets:

```text
bailA
germanA
pokec
syn
```

## Input Format

Each method/dataset pair has one directory:

```text
visualization/embeddings/<method>/<dataset>/
```

Place these two files in the directory:

```text
feat.npz
labels.npz
```

The default key in `feat.npz` is `representations`, with shape `[num_nodes, embedding_dim]`. The default key in `labels.npz` is `labels`, with shape `[num_nodes]`.

The default four-class encoding is:

```text
0 -> Y=1, S=0
1 -> Y=1, S=1
2 -> Y=0, S=0
3 -> Y=0, S=1
```

If a method exports `y` and `sens` separately, place them in either archive. The reader automatically encodes the pair into the four groups above. Supported key aliases are:

```text
y / labels_y / target / targets
sens / s / sensitive / sens_labels
```

## Output Format

Each `{method}_{dataset}` pair receives its own directory:

```text
visualization/results/<method>_<dataset>/
```

The regular mode writes:

```text
coordinates.csv
<method>.png
<method>.pdf
```

For example:

```text
visualization/results/SFFGNN_pokec/coordinates.csv
visualization/results/SFFGNN_pokec/SFFGNN.png
visualization/results/SFFGNN_pokec/SFFGNN.pdf
```

Figures use the paper style shared with the CELL/Pilot reference: a `6.4 x 4.35` inch canvas, coordinates normalized to 0.0--1.0, ticks from 0.0 to 1.0 at 0.2 intervals, and a complete gray frame. Tick labels use 18 pt and the bottom method/dataset label uses 20 pt. Scatter points use 20 pt². Four enlarged single-row legends sit immediately above the frame and are constrained to its width. Output uses `bbox_inches="tight"` with 0.1 inch padding.

The bottom label has this form:

```text
<method> t-SNE result on <dataset>
```

`<dataset>` comes from the `name` field in `config.yaml` and is also used in input and output paths.

## Usage

Check dependencies:

```bash
python visualization/run_tsne.py --check-deps
```

List configured methods and datasets:

```bash
python visualization/run_tsne.py --list
```

Run enabled methods and datasets from `config.yaml`:

```bash
python visualization/run_tsne.py
```

Select methods and datasets explicitly:

```bash
python visualization/run_tsne.py --datasets pokec --methods SFFGNN,DGSDA,HGDA
```

Generate several datasets at once:

```bash
python visualization/run_tsne.py --datasets bailA,germanA,pokec,syn --methods SFFGNN
```

Set the maximum number of sampled points:

```bash
python visualization/run_tsne.py --datasets pokec --methods SFFGNN --max-points 3000
```

Try several node-sampling seeds for a fixed embedding:

```bash
python visualization/run_tsne.py --datasets pokec --methods EMBER --sample-seeds 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 --tsne-seeds 0 --strict
```

Passing multiple `--sample-seeds` or `--tsne-seeds` enables batch candidate mode. This mode writes PNG files only, without PDF files or `coordinates.csv`; all candidates are placed in one directory:

```text
visualization/results/EMBER_pokec/candidates/EMBER_sample_seed_<sample_seed>_tsne_seed_<tsne_seed>.png
```

Batch mode ignores `--formats` and `plot.formats` in the configuration. `sample_seed` controls which embedding rows are sampled, while `tsne_seed` controls the t-SNE random state. Sampling is stratified and without replacement. With one seed, regular mode writes `coordinates.csv`, PNG, and PDF. For example, after selecting `sample_seed=17`:

```bash
python visualization/run_tsne.py --datasets pokec --methods EMBER --sample-seeds 17 --tsne-seeds 0 --strict
```

When comparing methods, use the same sampling seed for all methods and verify that their `coordinates.csv` files contain the same `source_index` values:

```bash
python visualization/run_tsne.py --datasets pokec --methods DANCE,GraphCTA,EMBER --sample-seeds 17 --tsne-seeds 0 --strict
```

Use `--strict` to fail immediately when an input file is missing:

```bash
python visualization/run_tsne.py --datasets pokec --methods SFFGNN,DGSDA --strict
```

## Export Helper

SFFGNN or a baseline can reuse `export_utils.py`:

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

The target-node range is the union of train, validation, and test splits, excluding nodes with label `-1` in Pokec. Compute representations on the complete target graph first, then apply the same `all_mask` to `feature`, `y`, and `sens`.

### Export From Baselines

The five SFDA runners support direct export. The commands below run the regular multi-run experiments, but only `run_idx=0` writes to the visualization directory:

```bash
python baseline/DGSDA/dgsda_sf.py --save_visualization_embeddings --visualization_run_idx 0
python baseline/GRADE/grade_sf.py --save_visualization_embeddings --visualization_run_idx 0
python baseline/HGDA/hgda_sf.py --save_visualization_embeddings --visualization_run_idx 0
python baseline/UDAGCN/udagcn_sf.py --save_visualization_embeddings --visualization_run_idx 0
python baseline/GraphAny/run_sfda.py --save_visualization_embeddings --visualization_run_idx 0
```

The final target representations exported by the methods are:

```text
DGSDA    -> hidden representation from the target Bern filter
GRADE    -> final-layer GCN representation before the classifier
HGDA     -> weighted combination of the three spectral branches
UDAGCN   -> fused GCN/PPMI attention representation
GraphAny -> attention-weighted representation before channel summation
```

All outputs use:

```text
visualization/embeddings/<method>/<dataset>/feat.npz
visualization/embeddings/<method>/<dataset>/labels.npz
```

For example:

```text
visualization/embeddings/SFFGNN/<dataset>/feat.npz
visualization/embeddings/SFFGNN/<dataset>/labels.npz
```

The current convention does not distinguish `source_trained` from `adapted`. If a different representation is required, ensure manually that `feat.npz` and `labels.npz` contain the corresponding stage.

## Adding a Baseline

Prepare the standard embedding directory:

```text
visualization/embeddings/<NewMethod>/<dataset>/feat.npz
visualization/embeddings/<NewMethod>/<dataset>/labels.npz
```

Then add the method to `config.yaml`:

```yaml
  - name: NewMethod
    enabled: true
```

If the archive keys are not `representations` and `labels`, configure them for that method:

```yaml
  - name: NewMethod
    enabled: true
    embedding_key: z
    label_key: group
```

Custom paths are also supported:

```yaml
  - name: NewMethod
    enabled: true
    embedding_paths:
      - "{project_root}/somewhere/{method}/{dataset}/embedding.npz"
    label_paths:
      - "{project_root}/somewhere/{method}/{dataset}/label.npz"
```

Available placeholders:

```text
{project_root}
{visualization_root}
{embeddings_root}
{results_root}
{method}
{dataset}
```
