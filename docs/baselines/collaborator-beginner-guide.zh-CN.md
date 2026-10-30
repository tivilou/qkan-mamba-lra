# QKAN-Mamba 实验运行步骤

## 1. 创建 GPU 服务器

选择以下环境，并将项目目录和数据目录放在平台提供的持久化磁盘上：

| 配置 | 选择 |
|---|---|
| 系统 | Ubuntu 22.04，x86_64 |
| Python | 3.11，镜像内包含 Miniconda 或 Anaconda |
| CUDA Toolkit | 12.6，选择带 `nvcc` 的 `devel` 开发镜像 |
| NVIDIA 驱动 | 560.28.03 或更新的兼容驱动 |
| GPU 显存 | 建议 24GB 或以上 |
| 内存 | 建议 32GB 或以上 |
| 磁盘可用空间 | 建议 50GB 或以上 |

## 2. 登录服务器

在本机终端执行平台给出的 SSH 命令，将用户名、服务器地址和端口替换为实际值：

```bash
ssh 用户名@服务器地址 -p SSH端口
```

下面各步在服务器终端执行；上传证书时另用本机终端。

## 3. 安装基础工具

```bash
sudo apt-get update
sudo apt-get install -y git build-essential tmux ca-certificates
```

使用 `root` 登录时，去掉两条命令前面的 `sudo`。

```bash
nvidia-smi
nvcc --version
```

## 4. 创建后台会话

```bash
tmux new -s qkan
```

在这个会话中继续执行后续命令。

## 5. 创建 Python 环境

```bash
conda create -n qkan-launch python=3.11 -y
conda activate qkan-launch

export QKAN_PYTHON="$(command -v python)"
export MAMBA_KEEP_CUDA_BUILD=TRUE
```

项目脚本会创建 `.venv-baselines`，自动安装 PyTorch 2.7.1 + cu126、NumPy、einops 和固定版本的 Mamba。

## 6. 从 GitHub 获取项目

```bash
mkdir -p "$HOME/projects"
cd "$HOME/projects"

git clone --branch research/collaboration-foundation --single-branch \
  https://github.com/tivilou/qkan-mamba-lra.git

cd qkan-mamba-lra
```

GitHub 要求认证时，输入自己的 GitHub 用户名，并使用 GitHub token 作为密码。

## 7. 配置文件交换服务

向研究方取得服务地址、项目访问 token 和 `exchange-ca.crt` 证书。

在服务器上创建证书目录：

```bash
mkdir -p "$HOME/qkan-private"
chmod 700 "$HOME/qkan-private"
```

在本机另开一个终端，将收到的证书上传到服务器。替换端口、用户名、地址和本机证书路径：

```bash
scp -P SSH端口 "本机证书文件的完整路径" 用户名@服务器地址:qkan-private/exchange-ca.crt
```

回到服务器的 tmux 会话，将下面的服务地址替换为研究方提供的完整地址：

```bash
export PROJECT_EXCHANGE_URL='研究方提供的完整HTTPS地址'
export PROJECT_EXCHANGE_CA_FILE="$HOME/qkan-private/exchange-ca.crt"

read -rsp '请输入项目访问 token: ' PROJECT_EXCHANGE_TOKEN
echo
export PROJECT_EXCHANGE_TOKEN
```

输入 token 后按回车。输入过程中终端不显示字符。

## 8. 启动 GPU 预检并自动获取数据

设置数据缓存目录：

```bash
export QKAN_CACHE="$HOME/qkan-cache"
mkdir -p "$QKAN_CACHE"
```

将 `--gpus 0` 中的 `0` 替换为平台分配的 GPU 编号，然后运行：

```bash
bash scripts/run_baselines.sh \
  --mode preflight \
  --cache "$QKAN_CACHE" \
  --gpus 0 \
  --models transformer,s4d_minimal,mamba1 \
  --tasks listops,text,retrieval,image,pathfinder \
  --seeds 42
```

脚本会自动安装依赖、获取上游代码、从作者地址下载数据、构建数据集，再执行 GPU 预检。已经准备好的内容会复用。

运行结束后，将终端显示的 `Suite: ...` 路径发给研究方。

## 9. 收到正式运行通知后启动实验

研究方提供正式实验版本和启动参数后，在项目目录更新代码：

```bash
git pull --ff-only origin research/collaboration-foundation
```

按通知中的模型、GPU 和种子参数执行。单卡、种子 `42` 的命令形式如下：

```bash
bash scripts/run_baselines.sh \
  --mode run \
  --cache "$QKAN_CACHE" \
  --gpus 0 \
  --models transformer,s4d_minimal,mamba1 \
  --tasks listops,text,retrieval,image,pathfinder \
  --seeds 42
```

这条命令会自动完成数据获取、GPU 预检、训练、评估和结果上传。

## 10. 保持任务运行

按 `Ctrl+B`，松开，再按 `D`，离开 tmux 会话。

重新登录服务器后，执行以下命令返回：

```bash
tmux attach -t qkan
```

## 11. 等待结果上传完成

训练结束后，脚本会自动将配置、指标、模型权重和恢复文件上传到项目文件服务。

实验文件保存在项目的 `runs/baseline-.../` 目录。脚本完成上传后会在该目录生成 `delivery-complete.json`。
