# Feature Topology Alignment for Traffic Risk Prediction

Research code for **Feature Topology Alignment Shapes Posterior Geometry in Foundation Models for Multimodal Traffic Risk Prediction**.

Zhenduo Zhang, Xiao Han, Yiling Wu, Xinfeng Zhang, Qingming Huang  
University of Chinese Academy of Sciences · Pengcheng Laboratory  
Correspondence: xfzhang@ucas.ac.cn

[中文说明](README_zh-CN.md) · [Experiment protocols](docs/experiments.md) · [Data format](docs/data.md) · [Reproduction notes](docs/reproducibility.md)

This repository compares Raw, One-Hot and Periodic representations of traffic features using TabPFN v2.5. It includes balanced-context inference, prior-probability correction, component ablations, conventional and modern tabular baselines, coordinate-group holdouts, calibration analysis, probability-space geometry and resource measurements.

## Installation

Use Python 3.11 in a separate environment. Run all commands from the repository root. The main TabPFN and spatial experiments use CUDA; the synthetic example and result analysis run on CPU.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,plots]"
python main.py --help
```

Install only the backends needed for the selected experiment. Install a PyTorch build appropriate for your GPU using the [official PyTorch instructions](https://pytorch.org/get-started/locally/) before adding GPU backends.

```bash
python -m pip install -e ".[tabpfn]"
python -m pip install -e ".[modern]"
python -m pip install -e ".[boosting]"
# CARTE has an additional graph/text environment; see docs/dependencies.md.
python -m pip install -e ".[carte]"
```

TabPFN is pinned to package `6.0.6` and explicitly selects checkpoint family `v2.5`. TabICL explicitly selects `tabicl-classifier-v1-20250208.ckpt`. Obtain pretrained weights through the upstream packages and follow their access terms. Model files are kept outside version control.

## Quick CPU example

```bash
python main.py demo --output-dir runs/demo
python main.py summarize --input runs/demo/results --output-dir runs/demo-summary
python -m pytest
```

The demo generates artificial data and trains Logistic Regression with Raw and Periodic inputs. Its metrics demonstrate the pipeline and are not manuscript results. No licensed records or model downloads are needed.

## Data

Place an authorized model-ready dataset at `data/local/SAICD.csv`. For the spatial experiment, also supply the original location annotation file `data/local/bz.csv` with its row order intact.

```text
location_id,road_type_code,hour,is_holiday,rainfall,temperature,wind,visibility,label
```

`hour` is an integer in 0–23; `label` is binary. See [data.md](docs/data.md) for field definitions and the location mapping rule.

```bash
python main.py validate --data data/local/SAICD.csv
```

SEWTRD is subject to a commercial data agreement and is not redistributed here. The model-ready CSV used for the recorded experiments contains 18,827 rows and 1,280 positives. Dataset access and preparation boundaries are described in [data.md](docs/data.md).

## Run the experiments

Each configuration records its data path, output path, seeds and model choices. Inspect a resolved configuration before launching a model:

```bash
python main.py --config configs/iid.json --dry-run
python main.py --config configs/iid.json
python main.py --config configs/ablation.json
python main.py --config configs/xgboost.json
python main.py --config configs/modern.json
python main.py --config configs/carte.json
python main.py --config configs/classical.json
python main.py --config configs/spatial_prepare.json
python main.py --config configs/spatial.json
```

Append command options to override a configuration, for example:

```bash
python main.py --config configs/iid.json --seeds 42 --encodings periodic --output-dir runs/iid-one-seed
python main.py --config configs/modern.json --models tabm --seeds 11 --output-dir runs/tabm-one-seed
```

| Experiment | Models / conditions | Train / validation / test | Recorded release seeds |
|---|---|---|---|
| `iid` | Raw, One-Hot, Periodic; balanced support + prior correction + validation F1 threshold | stratified 70/10/20 | 42–46 |
| `ablation` | 3 encodings × 2 support modes × 2 correction choices × 2 threshold rules = 24 cells | same IID split | 42–46 |
| `xgboost` | fixed XGBoost Raw/Periodic control | same IID split | 42–46 |
| `modern` | FT-Transformer, TabM-mini, TabICL-v1; Raw/Periodic | stratified 64/16/20 | 11,22,33,44,55,66,77,88,99,110 |
| `carte` | frozen-pretraining, single-model CARTE pilot; Raw/Periodic | same 64/16/20 split | same 10 seeds |
| `spatial` | TabPFN Raw/One-Hot/Periodic in G/S families | coordinate-disjoint, target 70/15/15 | 43,45,46,48,51 |
| `classical` | LR, MLP, Random Forest, XGBoost, LightGBM, CatBoost | release 70/10/20 protocol | 42–46 |

The conventional baseline runner supplies explicit common-protocol settings. Historical baseline metrics are archived separately; their original training scripts and complete hyperparameters were not recovered. The [protocol guide](docs/experiments.md) distinguishes these outputs.

## Results and figures

```bash
python main.py summarize --input runs/ablation --output-dir runs/ablation-summary
python main.py summarize --input runs/ablation --where SupportMode=balanced PriorCorrection=true ThresholdMode=validation_f1 --output-dir runs/iid-primary-summary
python main.py figures --input runs/iid --output-dir runs/iid-figures
python main.py reported --output-dir runs/reported-tables
```

`summarize` produces per-seed metrics, mean/sample-SD tables and paired encoding comparisons with 95% paired-t intervals. Seed IDs are matched before testing. Holm corrections apply across the comparisons in that invocation. Use `--left raw --right onehot` to change the contrast.

For a single seed, the summary retains the descriptive difference and leaves SD, confidence intervals and test p-values empty.

`figures` exports probability-margin distributions and reliability diagrams as PNG and PDF. Geometry is measured from the output vector `[1-p, p, 2p-1]`; this vector has one degree of freedom. The analysis describes output probabilities rather than internal transformer representations.

`results/reported/` contains recorded aggregate and per-seed metrics. The `reported` command recalculates the release tables without training any models. In the primary five-seed IID condition, mean F1 is 0.6212 for Raw, 0.6360 for One-Hot and 0.6407 for Periodic. These values belong to the stated release protocol. The stored results do not imply that the GPU experiments were rerun during code packaging.

## Repository layout

```text
main.py                  unified command entry point
fta/                     preprocessing, inference, experiments and analysis
configs/                 explicit experiment configurations
data/README.md           local data placement
docs/                    protocols, dependencies and provenance
results/reported/        recorded metrics, without sample-level records
tests/                   CPU checks for preprocessing and experiment wiring
.github/workflows/       CPU test workflow
CITATION.cff             manuscript author and title metadata
```

Run outputs contain local split indices and prediction arrays. `runs/`, licensed data and checkpoints are ignored by Git. Use a new output directory for each run; existing result tables are not silently overwritten.

## Citation and licensing

The manuscript title and authors are in [CITATION.cff](CITATION.cff). A DOI and publication URL can be added when available. This prepared release does not select a code license on the authors' behalf; the rights holder should add a license before granting reuse rights. Dependencies, pretrained weights and source data retain their own terms. See [upstream projects](docs/dependencies.md).
