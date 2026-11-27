# MEGA 一次性优化短测

统一入口 `diagnose mega-fft --all-methods` 一次测试当前已有环境支持的全部候选，最后统一回传一次，再恢复原训练／美化看板。实际速度、数值和支持情况以实验机结果为准。

| 单项 | 内容 |
|---|---|
| FFT | 双向 EMA FFT 长度补到 2 的幂，ListOps 8190→8192，保留原卷积截取 |
| BF16 | BF16 autocast，FP32 主参数和 Adam；检查误差，不直接准入 |
| TF32 | FP32 参数下开启 TF32，单独记录误差 |
| Graph | 现有 Torch 1.13 CUDA Graph API，仅包装临时模型实例 |
| micro-down / micro-up | 当前 micro 的一半／两倍，保持逻辑 batch；越界记失败，不改变正式配置 |
| resident | 临时将同一待训练 batch 驻留 GPU，测消除取数／传输的收益上界 |

此外无条件尝试 FFT＋BF16、FFT＋Graph、BF16＋Graph、FFT＋BF16＋Graph。单项中吞吐中位数提高至少 3%、4 对中至少 3 对更快的候选，自动追加一组兼容组合；上下 micro 只选收益更高者，BF16 与 TF32 不叠加，重复组合不再跑。组合筛选只代表速度候选，不代表数值准入。

每组独立进程，使用相同最新检查点、样本行和逻辑 batch。失败、OOM、当前环境不支持的 Graph 操作或数值非有限，均记入最终报告后继续其他组；进程清理未确认时停止后续派发。稳定训练核心、共享上游文件、正式 Suite、注意力分块和训练预算保持原样。使用已有 `.venv-mega`，不升级或安装依赖。现有环境没有 Torch 2 compile／SDPA，不把升级环境伪装成一个已测试开关。

## 当前推荐：Graph 捕获修复（2026-11-27）

使用 `--graph-repair --job auto`。上一轮 Text 的尾层 BF16 方案保留为候选，本轮不重跑全模型 BF16、TF32、micro 和驻留数据测试。固定6组：

1. 原 FP32＋保留mask＋EMA常量索引缓存对照。
2. 同一修复后的整模型 Graph。
3. FFT＋整模型 Graph。
4. FFT＋尾层 FFN BF16＋整模型 Graph。
5. 仅最后一层 FFN 的局部 Graph，EMA／FFT保持 eager。
6. FFT＋尾层 FFN BF16＋同一层局部 Graph。

固定官方 EMA 的 `_compute_kernel` 每次执行 `torch.arange(length).to(p)`，存在捕获时CPU取值／传输风险。新实例插件只缓存同长度、device、dtype的常量索引，首次在捕获外准备；可训练参数、卷积核和梯度每次重算。纯FP32对照用于检验此边界的数值影响。局部Graph封装原FFN前向和同一组Parameter对象，关闭时仍执行原前向，不改共享源码或正式worker。

每组独立进程，11次临时更新，最多66次／0次正式更新；失败隔离、统一回传、原训练与美化看板自动续跑沿用。整模型和局部Graph均以原生成功、实际replay计数和数值结果为依据，不把静态修复当作已跑通；无新增误差门槛或正式策略变更。失败回传增加Torch／fairseq文件和行号，不包含错误消息、输入、局部变量或私人路径。

## 历史轮次：选择性 BF16＋Graph 修复（2026-11-26）

第一轮已统一回传：FFT＋BF16 约 +16.85%，但更新向量相对差异约24%；四组 Graph 捕获失败。本轮使用 `--refine`，不重跑 micro、resident 和 TF32。

固定9组：FFT＋最后一层 FFN BF16、FFT＋全部 FFN BF16、FFT＋FFN/输出投影 BF16、纯 FP32 mask 修复对照、修复 Graph、FFT＋修复 Graph、BF16＋修复 Graph、FFT＋BF16＋修复 Graph、FFT＋FFN BF16＋修复 Graph。

选择性策略只在指定 Linear 内 autocast，输出立即转回 FP32；归一化、残差、门控、EMA、注意力分数和分类头不启用整体 autocast。尾层档只影响最后一层 FFN，全部 FFN 档和增加 h_proj 档逐步扩大范围。FP32 主参数、Adam、逻辑/micro batch不变。

Graph 插件校验官方编码器哈希，只在临时实例中移除 `padding_mask.any()` 的 GPU→Python 真值分支，保留布尔mask和原输出结构，不切换 traceable、不改共享文件。纯FP32 mask对照检查此变更本身的数值影响。它解决一个已识别捕获阻断点，不保证排除所有原生捕获问题；失败按白名单返回捕获阶段、同步、未用参数、FFT、autocast、OOM等分类，不回传任意错误文本或输入。

共9个独立子进程、每组11次临时更新，最多99次临时更新、0次正式更新。统一回传后恢复原正式策略与美化看板；不会把“测完”当作数值合格。保留原要求，不新增或放宽误差门槛。单batch误差变小仍不等于多步或收敛通过。每组300秒超时上限沿用，失败清理确认后继续。

## 实验机执行

先在原训练终端按一次 **Ctrl-C**，等检查点保存、入口退出，再粘贴整段。不要与正式训练同时占用 GPU 0。已暂停则直接执行。

```bash
cd /root/projects/qkan-mamba-lra-adaptive-89353fc83f89 &&
bash scripts/prepare_adaptive_checkout.sh &&
MEGA_DIAG_REV=$(git rev-parse FETCH_HEAD) &&
MEGA_DIAG_ROOT="$(dirname "$PWD")/qkan-mamba-lra-adaptive-${MEGA_DIAG_REV:0:12}" &&
cd "$MEGA_DIAG_ROOT" &&
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
.venv-baselines/bin/python -B scripts/experiment.py diagnose mega-fft -- \
  --source-suite /root/projects/qkan-mamba-lra-adaptive-89353fc83f89/runs/mamba-fused-formal-20261124T061405Z-b57105a1 \
  --cache "${QKAN_CACHE:-/root/qkan-cache}" --gpu 0 --graph-repair --job auto &&
.venv-baselines/bin/python -B scripts/experiment.py \
  --root /root/projects/qkan-mamba-lra-adaptive-89353fc83f89 resume \
  --suite runs/mamba-fused-formal-20261124T061405Z-b57105a1 \
  --task-order retrieval-last
```

默认 `--job auto`：持有原 Suite 锁后读取当前状态，跳过已完成、有 summary 或没有 latest 指针的任务，在有完整断点的未完成 MEGA 中优先选择非 Retrieval 和已暂停任务。选择结果打印并写入 request／回传报告，最终由原检查点校验确认完整性；不回到历史固定步数，不启动未开始任务做诊断。可显式给 `--job mega-text-s42` 等覆盖，但已完成或无断点会清楚报错。若全部 MEGA 已完成或均无待训练断点，打印诊断跳过，随后仍接续原训练／看板；该分支没有测速报告，也不声称已测。

这是一个入口串联短测和原训练：短测打印阶段心跳，完成全部候选尝试并验证回传后恢复原训练、美化看板及 Retrieval 最后顺序，即使部分或全部候选失败。单项失败保留类型、代码位置和已有部分数值／计时。源状态异常、清理未确认、用户中断或回传失败时停在原暂停状态，保留报告供排查；再次续跑使用[原统一入口](experiment.md)。

## 结果与边界

- 历史 `--all-methods` 最多 12 个顺序原生子进程（当前 `--refine` 为9组/99次），每组最多 300 秒、单阶段无进展最多 180 秒；这些是超时上限，不是预计时长。
- 取最新完整断点的一个待训练逻辑 batch。每次恢复独立克隆的模型、Adam、RNG 和样本调度状态；独立检查 Adam step 起点及增加量，避免可变引用污染对照。
- 3 次数值观测兼预热，随后 ABBA×2 的 8 次计时，每组共 **11 次临时更新**，最多 **132 次临时更新、0 次正式更新**。数值比较包括参数、更新量、裁剪后梯度、Adam 状态及 RNG／调度一致性，并提供原策略重复误差作参照。
- 常规计时涵盖 CPU 取数、H2D、前后向、裁剪、Adam、样本调度和同步；resident 候选不计取数／传输，单独标注为重复 batch 上界。恢复、数值副本和 Graph 首次建图不计入稳态计时，首次观测另报。改变 micro 可能改变 dropout 随机轨迹，误差记录不解释为纯浮点噪声。本次不承诺整轮加速或收敛等价。
- 记录峰值显存、实际 FFT 调用和长度。结束时精确恢复临时状态，移除自有检查点副本，并验证原文件哈希保持一致。
- `runs/mega-fft-*/mega-fft.json` 自动回传到 `qkan-mamba/mega-fft/<运行名>`，沿用现有 SHA-256 回执工具。回传失败可重传同一报告，无须重做测速。

收到结果后先看完整更新吞吐，再看候选误差与原策略重复噪声。短测不打开正式策略；如有明确收益，再决定是否做实际保存／恢复验证。无需重跑其他模型的全量检查。
