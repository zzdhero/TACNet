# 交通风险预测中的特征拓扑对齐

论文 **Feature Topology Alignment Shapes Posterior Geometry in Foundation Models for Multimodal Traffic Risk Prediction** 的研究代码。

作者：Zhenduo Zhang、Xiao Han、Yiling Wu、Xinfeng Zhang、Qingming Huang。通讯邮箱：xfzhang@ucas.ac.cn。

项目包含主实验、24种组件消融、XGBoost 编码对照、FT-Transformer / TabM / TabICL / CARTE 基线、G/S 道路族内的坐标分组留出、校准与概率空间分析。完整参数见 [实验说明](docs/experiments.md)。

## 安装与试运行

使用 Python 3.11，在仓库根目录运行：

```bash
python -m venv .venv
# Windows PowerShell：.venv\Scripts\Activate.ps1
# Linux/macOS：source .venv/bin/activate
python -m pip install -e ".[dev,plots]"
python main.py demo --output-dir runs/demo
python -m pytest
```

示例会生成合成数据，并使用逻辑回归跑通 Raw / Periodic 流程。示例指标不属于论文实验结果，也不使用真实交通记录。

按实验安装模型依赖：

```bash
python -m pip install -e ".[tabpfn]"
python -m pip install -e ".[modern]"
python -m pip install -e ".[boosting]"
```

主实验与空间实验使用 CUDA。PyTorch 的安装方式按本机 GPU 环境选择；CARTE 另需图计算依赖、fastText 文件与预训练权重，见 [依赖说明](docs/dependencies.md)。

## 放置数据

将获授权的模型输入表放在 `data/local/SAICD.csv`。空间实验还需 `data/local/bz.csv`，必须保留原始行序。字段与映射规则见 [数据说明](docs/data.md)。真实数据、逐样本预测和模型权重未放入发布包。

```bash
python main.py validate --data data/local/SAICD.csv
python main.py --config configs/iid.json --dry-run
```

## 运行实验

```bash
python main.py --config configs/iid.json
python main.py --config configs/ablation.json
python main.py --config configs/xgboost.json
python main.py --config configs/modern.json
python main.py --config configs/carte.json
python main.py --config configs/classical.json
python main.py --config configs/spatial_prepare.json
python main.py --config configs/spatial.json
```

主实验使用三个编码、平衡支持集、先验修正和验证集 F1 阈值。消融实验覆盖：3种编码 × 2种支持集 × 2种先验修正设置 × 2种阈值规则，共24种条件。五次支持集抽样是同一训练集内的重复抽样与概率平均。

IID 主实验采用70/10/20划分；现代基线和 CARTE 采用64/16/20划分；空间实验在 G、S 道路族内分别留出坐标组，目标比例为70/15/15。各实验的划分、编码、随机种子均有单独记录，不能直接把不同协议的结果当作配对结果。

`classical` 提供 LR、MLP、Random Forest、XGBoost、LightGBM、CatBoost 的统一协议实现。已有历史基线指标也保存在仓库中；其原训练脚本及完整超参数未找回，因此新入口明确使用发布版配置。

可在配置后覆盖参数：

```bash
python main.py --config configs/iid.json --seeds 42 --encodings periodic --output-dir runs/one-seed
```

## 汇总与绘图

```bash
python main.py summarize --input runs/ablation --output-dir runs/ablation-summary
python main.py summarize --input runs/ablation --where SupportMode=balanced PriorCorrection=true ThresholdMode=validation_f1 --output-dir runs/primary-summary
python main.py figures --input runs/iid --output-dir runs/figures
python main.py reported --output-dir runs/reported-tables
```

汇总程序按种子匹配编码，输出均值、样本标准差、配对差值、95%区间及检验结果。绘图程序输出 margin 分布、可靠性图和相应 CSV。这里的几何分析基于模型输出概率，不把它表述为隐藏表征的直接测量。

`results/reported/` 保存已有的汇总与逐种子指标；`reported` 命令可以重算相应表格。主实验五个种子的平均 F1 为 Raw 0.6212、One-Hot 0.6360、Periodic 0.6407。此次整理未重新执行完整 GPU 训练。

## 上传 GitHub

上传仓库目录中的内容即可。`.gitignore` 已排除本地数据、运行结果、权重、缓存和环境文件。英文 README 是默认首页，中文 README 用于操作参考。

发布前由权利人选择代码许可证，并在有正式出版信息后更新 `CITATION.cff`。本次整理没有替作者授予开源许可。详细复现边界见 [reproducibility.md](docs/reproducibility.md)。
