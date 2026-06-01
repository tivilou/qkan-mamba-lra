# Experimenter Guide

This guide is for the experiment collaborator. It covers environment setup, how to run experiments, and how to submit results.

## Environment Setup

### Prerequisites

- Linux with NVIDIA GPU (CUDA 11.8+)
- Python 3.10+
- At least 16GB GPU memory recommended for Text task (seq_len=4096)

### Installation

```bash
git clone https://github.com/tivilou/qkan-mamba-lra.git
cd qkan-mamba-lra
pip install -r requirements.txt
```

If `mamba-ssm` fails to install, ensure you have the correct CUDA toolkit:
```bash
nvcc --version   # should be 11.8+
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install mamba-ssm causal-conv1d
```

### Data Preparation

**推荐方式：从 MEGA 项目下载（包含全部 5 个任务）**

```bash
# 下载（约 1.4GB）
wget https://dl.fbaipublicfiles.com/mega/data/lra.zip

# 解压
unzip lra.zip

# 转换为训练脚本需要的 .npz 格式
python scripts/convert_mega_lra.py --mega_dir ./lra --out_dir data/lra
```

转换完成后的数据结构：

| 任务 | 训练集 | 验证集 | 测试集 | 序列长度 |
|------|--------|--------|--------|---------|
| ListOps | 96,000 | 2,000 | 2,000 | 2048 |
| Text (IMDB) | 25,000 | 25,000 | 25,000 | 4000 |
| Retrieval (AAN) | 147,086 | 18,090 | 17,437 | 2000×2 |
| Image (CIFAR-10) | 45,000 | 5,000 | 10,000 | 1024 |
| Pathfinder | 160,000 | 20,000 | 20,000 | 1024 |

**备选方式：从 HuggingFace 分别下载（不需要翻墙）**

```bash
pip install datasets torchvision
python scripts/download_lra.py --out_dir data/lra
```

注意：备选方式中 Pathfinder 和 Retrieval 使用占位数据，仅用于调试。

**Output structure**:

```
data/lra/
  listops/    train.npz, val.npz, test.npz
  text/       train.npz, val.npz, test.npz
  retrieval/  train.npz, val.npz, test.npz
  image/      train.npz, val.npz, test.npz
  pathfinder/ train.npz, val.npz, test.npz
```

Each `.npz` file contains:
- `inputs`: numpy array of shape `(N, seq_len)` or `(N, seq_len, d_input)`
- `labels`: numpy array of shape `(N,)`
- For retrieval: `inputs1`, `inputs2`, `labels`

You can set a custom data path via environment variable:
```bash
export LRA_DATA_DIR=/path/to/your/lra/data
```

## Running Experiments

### Quick Test (single task, command line args)

适合快速验证某个任务是否能跑通、loss 是否下降。

```bash
python scripts/quick_train.py --task <task> --data_dir data/lra [options]
```

参数说明：
- `--task`: listops / text / retrieval / image / pathfinder
- `--data_dir`: 数据目录（默认 data/lra）
- `--epochs`: 训练轮数（默认 10）
- `--batch_size`: 批大小（默认 32）
- `--lr`: 学习率（默认 0.001）
- `--d_model`: 模型维度（默认 128）
- `--n_layers`: 层数（默认 4）
- `--seed`: 随机种子（默认 42）

各任务推荐配置：

```bash
# Image (收敛最快，适合验证环境)
python scripts/quick_train.py --task image --data_dir data/lra --epochs 15 --batch_size 64

# ListOps
python scripts/quick_train.py --task listops --data_dir data/lra --epochs 15 --batch_size 64

# Text (收敛慢，需要更多 epoch)
python scripts/quick_train.py --task text --data_dir data/lra --epochs 50 --batch_size 32

# Pathfinder (收敛慢)
python scripts/quick_train.py --task pathfinder --data_dir data/lra --epochs 50 --batch_size 128

# Retrieval (数据量大，序列最长，batch_size 要小)
python scripts/quick_train.py --task retrieval --data_dir data/lra --epochs 30 --batch_size 16
```

### Single Task (YAML config)

正式实验使用 YAML 配置文件，结果自动保存为 JSON。

```bash
python experiments/lra/train.py --config experiments/lra/configs/text_mamba_only.yaml
```

配置文件中修改 `task` 字段即可切换任务，修改 `model.name` 切换模型。

各任务 × 模型的推荐配置：

| 任务 | model.name | batch_size | epochs | lr | 备注 |
|------|-----------|-----------|--------|------|------|
| listops | mamba_only | 64 | 50 | 0.001 | 10 分类 |
| listops | qkan_mamba | 32 | 50 | 0.001 | |
| text | mamba_only | 32 | 50 | 0.001 | 收敛慢 |
| text | qkan_mamba | 16 | 50 | 0.001 | 序列长 4096 |
| retrieval | mamba_only | 16 | 30 | 0.001 | 序列 4096×2 |
| retrieval | qkan_mamba | 8 | 30 | 0.001 | 显存占用大 |
| image | mamba_only | 64 | 50 | 0.001 | 收敛快 |
| image | qkan_mamba | 32 | 50 | 0.001 | |
| pathfinder | mamba_only | 128 | 50 | 0.001 | 收敛慢 |
| pathfinder | qkan_mamba | 64 | 50 | 0.001 | |

### Override Seed

```bash
python experiments/lra/train.py --config experiments/lra/configs/text.yaml --seed 7
```

### Run All Tasks (5 seeds each)

```bash
bash scripts/run_all.sh qkan_mamba 0 4
```

### Run Baselines

To run a baseline model, modify the config's `model.name` field or create a new config:

```yaml
model:
  name: transformer  # or: s4, mamba_only
  params:
    d_model: 128
    n_layers: 4
    n_heads: 4
    d_ff: 256
    dropout: 0.1
    pooling: mean
```

Available models: `qkan_mamba`, `transformer`, `s4`, `mamba_only`

## Experiment Plan

We need results for 4 experiments:

### Exp 1: LRA Benchmark (main table)
Run all 5 tasks x 4 models x 5 seeds = 100 runs.

```bash
for MODEL in qkan_mamba transformer s4 mamba_only; do
    bash scripts/run_all.sh $MODEL 0 4
done
```

### Exp 2: Gate Ablation
Compare gate types on all tasks. Configs will be provided in `experiments/lra/configs/ablation/`.

### Exp 3: Sequence Length Sensitivity
Text task with varying max_len (1024, 2048, 4096, 8192). Configs in `experiments/lra/configs/seqlen/`.

### Exp 4: Parameter Efficiency
Fixed parameter budget comparison. Configs in `experiments/lra/configs/efficiency/`.

## Submitting Results

### Result Format

Each run produces a JSON file in `results/`:
```json
{
  "task": "listops",
  "model": "qkan_mamba",
  "seed": 42,
  "n_params": 123456,
  "best_val_acc": 0.58,
  "test_acc": 0.57,
  "history": [...]
}
```

### Git Workflow

```bash
# Before starting, sync with main
git pull origin main

# Create an experiment branch
git checkout -b exp/lra-listops-qkan

# After experiments complete
git add results/
git commit -m "exp: ListOps QKAN-Mamba results (seed 0-4)"
git push -u origin exp/lra-listops-qkan

# Open a PR for review
gh pr create --title "Exp: ListOps QKAN-Mamba results" --body "5-seed results for ListOps task"
```

### Naming Convention

Branch: `exp/<task>-<model>` (e.g., `exp/lra-text-transformer`)
Result files: `results/<task>/<task>_<model>_s<seed>.json`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `batch_size` in config |
| mamba-ssm install fails | Check CUDA version matches PyTorch |
| Data not found | Set `LRA_DATA_DIR` env variable |
| NaN loss | Reduce `lr` to 5e-4 |

## Communication

- Open a GitHub Issue for bugs or questions
- Tag results PRs with the experiment number (Exp 1/2/3/4)
- If a run crashes, note the error in the PR description
