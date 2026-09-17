<p align="center">
  <img src="docs/imgs/logo.png" width="480" alt="PocketWAM logo">
</p>
<p align="center"><h1 align="center">PocketWAM：Building a 0.003B World Action Model from Absolute Zero</h1></p>
<p align="center">简体中文 | <a href="docs/README_EN.md">English</a></p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="python">
  <img src="https://img.shields.io/badge/torch-2.7%2B-EE4C2C?logo=pytorch&logoColor=white" alt="torch">
  <img src="https://img.shields.io/badge/transformers-4.51%2B-FFD21E?logo=huggingface&logoColor=white" alt="transformers">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="license">
</p>

## 项目介绍

🚀 **继开源 [PocketVLA](https://github.com/penpenzh/pocketvla)**（如果不熟悉本项目子任务建议请先参考：[PocketVLA](https://github.com/penpenzh/pocketvla)）**之后，
又开源了 —— PocketWAM**：依旧**完全从零开始**（极其轻量仿真环境、数据生成、模型、两阶段训练、
闭环评测、交互演示全部从零实现），搭建一个参数量仅 **0.003B** 的
**WAM（World Action Model，世界-动作模型）**，完整实现**联合生成未来世界状态与动作**的范式，
并端到端验证其在闭环控制任务上的有效性：在同一子任务上，PocketWAM 以**纯SFT**
取得 **100%** 闭环成功率，超过了 PocketVLA 在低数据量示教上训练的 SFT 版本（74.2%），
也高于其进一步用 GRPO 强化学习后的版本（80.2%）。

🧠 **WAM 与 VLA 的核心区别**：VLA 只学习「状态 → 动作」；WAM 则**联合生成「未来世界状态 + 当前动作」**，用想象出的终态图像作为隐式视觉规划器来引导动作生成：

```
π(o_goal, a | o_t, c, q_t)  =  π(o_goal | o_t, c)        ·  π(a | o_t, o_goal, q_t)
                                └─ 视频预测(世界模型)        └─ 逆动力学(动作)
```

🦾 在 2 连杆机械臂「语言引导抓取-放置」任务上，模型每步**只输出两样东西**：**想象终态图像**
（64×64“梦境”：指令球已入筐）+ **当前动作** `(Δq1, Δq2, grip)`，无任何辅助头；推理为两遍式 ——
Pass A 联合去噪出「梦境 + 动作草稿」，Pass B 在 t=1 拿着想象图**显式解码**动作
（⚡ Flash 单步模式 `denoise_steps: 1`，每步仅 2 次前向）。🎬 真实闭环回放 👇

<p align="center">
  <img src="docs/imgs/ep001.gif" width="47.5%" alt="ep001 closed-loop replay">
  <img src="docs/imgs/ep007.gif" width="47.5%" alt="ep007 closed-loop replay">
</p>
<p align="center"><sub>真实观测| Δ 变化图|想象终态|</sub></p>

📌 本项目**继续沿用与 PocketVLA 完全相同的 [2D-Grab 任务](https://github.com/penpenzh/pocketvla#%E4%B8%8B%E6%B8%B8%E4%BB%BB%E5%8A%A1-2d-grab)**：
仿真、评测协议与全部测试数据（固定测试集 500 场景、OOD 集 300 场景）；
🎯 目的是**以更轻量、更易读的实现，帮助读者快速入门 WAM 范式**。

**🎉本项目包含以下内容**：

- 🧪 **极轻量仿真环境**（`pwam/sim/`，与 PocketVLA 逐行一致）：2 连杆平面臂俯视 2D「抓取-放置」+
  64×64 观测渲染（不标记目标、无观测泄漏）+ GIF 回放。
- 🌍 **从零实现的 WAM 模型**（`pwam/model/wam.py`）：每一个模块都是从0搭建。
- 🎯 **WAM 范式数据管线**（`pwam/data/collect.py`）：可自行扩展、定义新任务等。
- 📚 **两阶段训练**（`pwam/train.py`）：Stage-0：video-only 世界模型预训练；Stage-1： SFT
  从预训练骨干初始化，拟合「视频（想象终态）+ 两遍式动作」联合损失；类平衡夹爪损失。
- 📊 **闭环评测**（`pwam/evaluate.py`）：与 PocketVLA 完全相同的协议与指标（闭环成功率 / 抓对率 / 落点
  距离 / 失败归因 / 分颜色统计 / 专家轨迹对比），外加 WAM 特有的**世界模型指标**（PSNR等）。
- 🌗 **OOD 泛化评测**（`pwam/eval_ood.py`）：300 场景纯净 OOD 协议——训练数据零干扰物暴露，
  测试时加入 8 种未见颜色的球/方块干扰物。

## 下载

**模型权重**

| 模型 | 参数量 | 阶段 | 闭环成功率 | 下载 |
|---|---|---|---|---|
| pocketwam-3m-pt-2d-grab | 3.23M | Stage-0 预训练 | —（video-only，动作头未训练） | 🤗 [HuggingFace](https://huggingface.co/penpenzh/pocketwam-3m-pt-2d-grab) |
| pocketwam-3m-pt-sft-2d-grab | 3.23M | 预训练 + SFT | **100.0%** | 🤗 [HuggingFace](https://huggingface.co/penpenzh/pocketwam-3m-pt-sft-2d-grab) |

**数据**

| 文件 | 规模 | 下载 |
|---|---|---|
| test.npz | 500 场景 | 🤗 [HuggingFace](https://huggingface.co/datasets/penpenzh/pocketvla-dataset-2D-grab/blob/main/test.npz) |
| ood_test.npz | 300 场景 | 🤗 [HuggingFace](https://huggingface.co/datasets/penpenzh/pocketvla-dataset-2D-grab/blob/main/ood_test.npz) |
| train.npz | ~14,000 回合 / ~35.9 万样本 | 脚本生成 |
| val.npz | ~6,000 回合 / ~15.4 万样本 | 脚本生成 |
| pretrain_synth.npz | 90,000 对 | 脚本生成 |
| pretrain_play.npz | ≈156,000 对 | 脚本生成 |

```bash
# 评测集：下载到 dataset/（evaluate / eval_ood 默认读取 dataset/test.npz 与 dataset/ood_test.npz）
huggingface-cli download penpenzh/pocketvla-dataset-2D-grab \
    test.npz ood_test.npz --repo-type dataset --local-dir dataset

# 训练集：不提供下载 —— 固定种子，一条命令完全复现
python3 -m pwam collect
```

## 快速开始

```bash
# 环境准备
conda create -n pocketwam python=3.10 -y
conda activate pocketwam
pip install -r requirements.txt
```

```bash
# Run the full pipeline (collect -> pretrain -> train -> evaluate -> OOD eval)
bash scripts/run_all.sh
```

或分步运行:

```bash
# 0. Dataset generation: SFT demos (with WAM expansion + final-state targets)
#    + Stage-0 pretrain set (synthetic goals + play dynamics)
#    + test/ood_test scenario sets (identical to PocketVLA) in one shot
python3 -m pwam collect

# 1. Stage 0: world-model pretraining (video-only) -> outputs_pretrain_wam_3m/
python3 -m pwam pretrain --config configs/sft_wam_3m.yaml

# 2. Stage 1: SFT, initialized from the pretrained backbone -> outputs_sft_wam_3m/train/
python3 -m pwam train --config configs/sft_wam_3m.yaml

# 2. Closed-loop eval on the fixed 500-scenario test set
#    (GIFs + failure attribution + figures + world-model PSNR) -> outputs_sft_wam_3m/eval/
python3 -m pwam evaluate --config configs/sft_wam_3m.yaml

# 3. OOD generalization eval (unseen-color distractor, 300 scenarios)
python3 -m pwam.eval_ood --config configs/sft_wam_3m.yaml

```

## 快速推理示例

```python
import pwam.model   # register PocketWAM with AutoModel/AutoTokenizer (run from repo root)
import torch
from transformers import AutoModel, AutoTokenizer

ckpt = "outputs_sft_wam_3m/train/pocketwam-3m"   # 或指向 HF 下载目录：penpenzh/pocketwam-pt-sft-2d-grab
model = AutoModel.from_pretrained(ckpt, trust_remote_code=True).eval()
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

image = torch.rand(1, 3, 64, 64)                    # top-down view in [0, 1]
enc = tok("put the green ball into the bin", padding="max_length",
          truncation=True, max_length=model.config.lang_len)
tokens = torch.tensor([enc["input_ids"]])
pad = tokens != 0
proprio = torch.tensor([[0.5, -0.3, 0.2, 0.4, 0.0, 0.0]])
# (q1/pi, q2/pi, ee_x/1.1, ee_y/1.1, gripper, holding)

with torch.no_grad():
    goal, action = model.generate(image, tokens, pad, proprio, steps=1)

print("imagined final state:", goal.shape)          # (1, 3, 64, 64) in [-1, 1]
print("action (dq1, dq2, grip):", action[0].tolist())
# 拿到动作后交给仿真环境闭环执行: obs, _, done, info = env.step(
#     [action[0,0]*0.15, action[0,1]*0.15, action[0,2]])
```

## PocketWAM 模型结构

<p align="center"><img src="docs/imgs/arch_wam.png" width="2400" alt="WAM architecture"></p>

**总览**：参数量 **0.003B（3.23M）**。输入 = `(观测 64×64×3, 指令 10 tokens, 本体感知 6 维)`，
输出 = `(想象终态图像 64×64×3, 动作 (Δq1, Δq2, grip))` 。
一个共享 trunk 负责全部联合建模，两个头分别解码「梦境」与「动作」。

### 🧱 网络组件

| 组件 | 结构（yaml 实配） | 输出 |
|---|---|---|
| 👁️ 视觉编码器 | 5 级 CNN：通道 24→48→72→96→96，前 3 级 stride-2（64→8），后 2 级 + 4 个 stride-1 块精炼；GroupNorm + SiLU | 8×8×96 特征图 → 64 个 vision tokens |
| 💬 指令编码 | 22 词小词表嵌入 + 位置嵌入 | 10 个 text tokens（唯一参与 padding 掩码的部分） |
| 🦾 本体编码 | Linear(6→192)；`(q1,q2)/π、(ee_x,ee_y)/1.1、夹爪、持球` 归一化 | 1 个 proprio token |
| 🎯 终态编码器 | 3 级 stride-2 CNN：3→48→80→192（64→8） | 噪声/中间终态图 → 64 个 goal tokens |
| 🕹️ 动作槽 | Linear(3→192) | 1 个 action token |
| 🧠 共享 trunk | Pre-LN 双向 Transformer × 6 层（d=192、4 heads、d_ff=448、SiLU-MLP）；仅指令 padding 掩码，其余全双向 | 140 tokens 的联合隐状态 |
| 🖌️ GoalDecoder | 读 [goal hidden ⊕ vision hidden]（两个 8×8 网格按通道堆叠 → 384 通道），3 次 (Upsample×2 + Conv) 上采样 8→64 | 残差 Δ̂ (3,64,64)，x̂1 = 观测 + Δ̂ |
| 🤖 ActionHead | MLP 192→640→160→3 + tanh | action token 隐状态 → 动作 (Δq1, Δq2, grip) |

**两个关键设计**（3M 也能生成清晰终态的原因）：

- **残差 Δ 参数化**：`x̂1 = 当前观测 + Δ̂` —— 生成器只需预测**世界的变化**（移动的手臂、入筐的球），
  观测中不变的 97% 像素零成本透传，全部生成容量集中在动力学上。
- **解码器交叉读取 vision tokens**：goal decoder 的输入 = [goal hidden ⊕ vision hidden]，世界模型损失
  **直接监督 vision tokens 的隐状态** —— 语言-颜色绑定在动作 query 能读到的位置成形。

### ⚡ 两遍式推理（Flash 单步模式，每控制步 2 次前向）

**x0-参数化**：两个头直接预测**干净目标** (x̂1, â1)，流速度场 `u = (x̂1−x_t)/(1−t)` 在推理时解析
恢复 —— 训练损失直接优化「清晰图像」本身；K 步 Euler 采样，最终配置 `denoise_steps: 1`（K=1）：

```
初始化    x = ε ~ N(0,I) ∈ (3,64,64)          a = ε_a ~ N(0,I) ∈ R³
Pass A    (x̂1, â1_draft) = forward(o_t, c, q_t, x, a, t=0)          # 共享 trunk 联合去噪（草稿）
Euler     x ← x + dt·(x̂1 − x)/(1 − t)      # K=1 时即一步到位；K>1 则沿流迭代
Pass B    â1_final = forward_idm(o_t, c, q_t, goal=x̂1, draft=â1_draft, t=1)   # 逆动力学精修
执行      action = (â1_final[:2] × 0.15 rad, â1_final[2])            # x̂1 仅用于展示与监控
```

- **Pass A**：干净上下文 [vision | text | proprio]（teacher forcing）与噪声生成块 [goal | action]
  （`x_t = t·x1+(1−t)·ε`，图像与动作**共享去噪时间步 t**）一起过 trunk，一遍前向同时出「梦境 + 动作草稿」。
- **Pass B（逆动力学，t=1）**：画好的 x̂1 重新经终态编码器回喂 trunk，动作在 t=1 处从估计的终态图像解码** —— 显式实现 `π(a | o_t, ô_goal, q_t)`：动作从具体的想象未来中进行预测。
- 闭环中**每步都重新想象**（新观测 → 新 Δ̂ → 新动作），等价于每步重新规划。

## 下游任务（2D-Grab）

📌 下游任务**直接沿用 [PocketVLA 的 2D-Grab 任务](https://github.com/penpenzh/pocketvla#%E4%B8%8B%E6%B8%B8%E4%BB%BB%E5%8A%A1-2d-grab)** —— 2 连杆平面机械臂在
俯视 2D 场景中的语言引导「抓取-放置」闭环任务；场景 / 指令 / 观测 / 动作 / 抓取 / 放置判据的
完整定义请见原项目的任务章节。仿真、解析专家与测试集生成逻辑与 PocketVLA 逐行相同：固定测试集
`test.npz`（500 场景，seed+20,000+i）、OOD 集 `ood_test.npz`（300 场景，seed+80,000+i）
**逐字节一致**，闭环评测协议（每步查询一次模型、执行单步动作、90 步内入筐即成功）完全相同，
结果严格可比。


## WAM数据生成

`python3 -m pwam collect` 一次生成全部 6 份数据。预训练集与 SFT 集种子空间完全隔离（+500,000 / +700,000 vs. seed=42 示教链）：

| 文件 | 规模 | 种子 | 说明 |
|---|---|---|---|
| `pretrain_synth.npz` | 90,000 对（Stage-0） | +500,000+i | 合成目标对，每场景渲染全部 3 种指令颜色的终态 |
| `pretrain_play.npz` | ≈156,000 对（Stage-0） | +700,000+i | play 动力学对 (o_t, a_t, o_{t+1}) |
| `train.npz` | ~14,000 回合 / ~35.9 万样本 | seed=42 链 | 示教 + 扩增（见下），含 `ep_final` |
| `val.npz` | ~6,000 回合 / ~15.4 万样本 | seed=42 链 | 按回合切分的验证集，同样含 `ep_final` |
| `test.npz` | 500 场景 | +20,000+i | **与 PocketVLA 一致** |
| `ood_test.npz` | 300 场景 | +80,000+i | **与 PocketVLA 逐一致** |


**Stage-0 两种成分**：

- `sg_*` 合成目标对（30,000 场景 × 3 指令颜色 = 90,000 对）：hindsight 合成终态——指令球传送至筐内释放点、手臂置于 IK 释放位形，直接渲染，无需任何轨迹。同一场景、三种指令、三种"梦境"，强迫世界模型真正读懂数据词与球的对应（对比式颜色绑定）。40% 场景先做 play/持球热启动，使起始状态含中间态。
- `pl_*` play 动力学对（12,000 回合 × 每回合 6~20 步随机动作 ≈ 156,000 对）：逐步记录 (o_t, a_t 已执行, o_{t+1})，训练时把已执行动作作为干净条件注入（10% classifier-free dropout），学习动作条件化的前向动力学——逆动力学动作头的物理基础。

**SFT 每步样本**：`(观测, 指令 token, 本体感知, 专家动作, 回合 id)`；另每回合一张 `ep_final` 最终状态图像（指令球在筐内）——WAM 从任意中间状态都要学会想象的世界模型监督目标。train/val 按回合切分（无帧泄漏）。

**Warm-start 扩增**（`warm_start_prob: 0.35`；扩增内 play 40% / carry_right 30% / carry_wrong 30%）：

| 类型 | 内容 | 动机 |
|---|---|---|
| `play` | 回合前随机预滚动 1~8 步（可能抓/拖错球），每个状态以专家动作重标注 | 异构、非重复的状态覆盖；错球由专家恢复 |
| `carry_right` | 从「已持有指令球」的位形开始 | 搬运相位的稠密覆盖 |
| `carry_wrong` | 从「持有错误球」开始，专家先松爪再抓对球 | 纯示教中从未出现的恢复行为，提升鲁棒性与 OOD |

所有扩增回合以成功终态收尾，`ep_final` 语义不变。

## 训练流程（两阶段）

**Stage 0 — video-only 世界模型预训练**



| 数据行 | action 槽输入 | 目标 x1 | 学到什么 |
|---|---|---|---|
| `sg_*` | 纯噪声（与 SFT Pass A 中「动作待生成」一致） | hindsight 渲染的终态 | 想象终态 + 对比式颜色绑定 |
| `pl_*` | 90% 真实执行的 a_t（干净 teacher-forcing）；10% 噪声（classifier-free dropout） | 真实下一帧 o_{t+1} | 动作条件前向动力学 `o_t + a_t → o_{t+1}` |

play 行的「做了这个动作，世界会变成什么样」正是 Stage 1 动作头（「世界要变成那样，此刻该做什么」）的物理基础：先懂因果，再学控制。

**Stage 1 — SFT**

每步样本采共享 `t ~ U(0,1)`，构造 `x_t = t·x1+(1−t)·ε`、`a_t = t·a1+(1−t)·ε_a`，两遍前向联合训练：

```
Pass A:  (x_t, a_t) ──共享trunk──> (x̂1, â1_draft)
Pass B:  (ô_goal, â1_draft) ──同一trunk @ t=1──> â1_final            # 逆动力学

L = w(t)·MSE(x̂1, x1) + 12·[ (dq_A + dq_B) + 0.3·w_grip·(grip_A + grip_B) ]

w(t)=(1−t)     压制高 t 的「去噪复制」捷径 → 逼出低 t 的语义生成
变化像素×10    
w_grip         (持球×标签) 4 组逆频率权重 → 防夹爪退化为永远张开/闭合
ô_goal         Pass B 以 50% 用真实 x1 / 50% 用自己的 x̂1（scheduled sampling，对齐部署分布）
```

## 实验结果

在固定 500 场景测试集上的闭环评测：

| 模型 | 范式| 闭环成功率 | 抓对球率 | 平均落点距离 | 成功回合步数 | 想象终态 PSNR |
|---|---|---|---|---|---|---|
| pocketvla-3m-grpo-2d-grab | [PocketVLA](https://github.com/penpenzh/pocketvla) | 80.2% | 83.4% | 0.220 | 29.4 | — |
| pocketwam-3m-sft-2d-grab| PocketWAM(SFT)| 25.6% | 33.2% | 0.760 | 35.9 | 17.4 dB |
| **pocketwam-3m-pt-sft-2d-grab** | **PocketWAM(PT+SFT)** | **100.0%** | **100.0%** | **0.055** | **25.7** | **22.6 dB** |


## 域外泛化测试 (OOD)

与 PocketVLA 完全相同的 300 场景 OOD 集（训练中从未出现的颜色干扰物 yellow/purple/orange/cyan/
pink/brown/white/gray，形状随机为球或方块；指令仍只指向三种训练过的颜色）：

（OOD 评测为**真实含干扰物版本**：每个场景多一个训练中从未出现的颜色干扰物——
yellow/purple/orange/cyan/pink/brown/white/gray 中随机一种，形状随机为球或方块，指令仍只指向
三种训练过的颜色。）

| 模型 | OOD 成功率 | 抓对球率 | 平均落点距离 | ID→OOD 落差 |
|---|---|---|---|---|
| pocketvla-3m-grpo-2d-grab | 36.0% | 67.0% | 0.472 | **−39.5 pt** |
| **pocketwam-3m-pt-sft-2d-grab**| **40.3%** | **87.7%** | 0.432 | −59.7 pt |


## 继续探索

> 💡 训练你自己的 `pocketwam-[x]m-[task]`。

## 引用

如果这个项目对你的学习或研究有所帮助，欢迎引用：

```bibtex
@misc{pocketwam2026,
  title  = {PocketWAM: Building a 3M World Action Model from Absolute Zero},
  author = {Enming Zhang},
  year   = {2026},
  url    = {https://github.com/penpenzh/pocketwam}
}
```

## License

本项目采用 **[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)** 协议开源（详见 LICENSE）
