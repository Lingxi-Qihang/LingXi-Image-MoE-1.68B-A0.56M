<div align="center">

# LingXi-Image-MoE: 从交替到全层<br>混合专家架构在文生图扩散模型中的对比研究<br><small>From Alternating to Full-Layer: A Comparative Study of Mixture-of-Experts Architectures for Text-to-Image Diffusion Models</small>

<a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank"><img src="https://img.shields.io/badge/论文-b5212f.svg?logo=arxiv" height="22px"></a>
<a href="https://huggingface.co/shyai/LingXi-Image-MoE" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20模型-d96902.svg" height="22px"></a>
<a href="https://github.com/Lingxi-Qihang/LingXi-Image-MoE" target="_blank"><img src="https://img.shields.io/badge/代码-181717.svg?logo=github" height="22px"></a>
<a href="https://modelscope.cn/models/haohanxingcheng/LingXi-Image-MoE" target="_blank"><img src="https://img.shields.io/badge/魔搭-624aff.svg" height="22px"></a>

</div>

---
### 本文所提及的“空间关系（Relation）全球第二”，是基于DPG-Bench公开评测基准跑出来的客观指标，并非我自封或夸大。DPG-Bench是业界公认的文生图评测标准之一，所有对比模型的分数均来自官方论文或公开榜单。我的1.68B模型已全面开源，任何研究者都可以下载权重、复现评测、验证这一结果。技术可以讨论，分数经得起检验。
---

###The "second highest spatial relationship in the world" mentioned in this article is an objective indicator based on the DPG Bench public evaluation benchmark, and is not self proclaimed or exaggerated by me. DPG Bench is one of the industry recognized standards for evaluating cultural and biological images, and the scores of all comparison models are obtained from official papers or public rankings. My 1.68B model has been fully open sourced, and any researcher can download weights, reproduce evaluations, and validate this result. Technology can be discussed, and scores can withstand scrutiny.

# 🔥 动态
- 🧠 **1.68B 模型权重**：[ModelScope LingXi-Image-MoE-1.68B-A0.56M ](https://modelscope.cn/models/haohanxingcheng/LingXi-Image-MoE/tree/master/LingXi-Image-MoE-1.68B-A0.56M)
- 📦 **125万步 checkpoint**：[ModelScope ckpt_step_1250000.pth](https://modelscope.cn/models/haohanxingcheng/LingXi-Image-MoE/tree/master/ckpt_step_1250000.pth)
- [2026-07-31] **ProMoE‑L (1.68B)** 联合训练完成，论文《能力涌现的时序规律：一个 1.68B 全层混合专家文生图模型训练的动力学分析》（Temporal Patterns of Capability Emergence: A Fine-Grained Analysis of Training Dynamics in a 1.68B 
- [2026-07-13] 论文《从交替到全层：混合专家架构在文生图扩散模型中的对比研究》（From Alternating to Full-Layer: A Comparative Study of Mixture-of-Experts Architectures for Text-to-Image Diffusion Models）发布在 arXiv。
- [2026-07-13] 模型权重同步上线 [HuggingFace](https://huggingface.co/shyai/LingXi-Image-MoE) 和 [ModelScope](https://modelscope.cn/models/haohanxingcheng/LingXi-Image-MoE)。
- [2026-07-12] 完整代码 [GitHub](https://github.com/Lingxi-Qihang/LingXi-Image-MoE) 开源。

> 0.47B 版本的训练和推理方案请参见 [LingXi-Image-MoE 主仓库](https://github.com/Lingxi-Qihang/LingXi-Image-MoE)。

---

## 论文简介

**《能力涌现的时序规律：一个1.68B全层混合专家文生图模型训练的动力学分析》**<br>
*Temporal Patterns of Capability Emergence: A Fine-Grained Analysis of Training Dynamics in a 1.68B Full-Layer MoE Text-to-Image Model*

大规模扩散模型的训练通常被视为一个"黑箱"。本文基于一个 **1.68B 参数的全层 MoE 文生图模型**，通过从 30 万步到 120 万步、每 5 万步一次的细粒度 DPG-Bench 评测，首次系统记录了五项核心能力的动态成长曲线，揭示了三大关键规律：

1. **早期定型**：空间关系能力在 <30 万步即已定型，长期稳定在 92–94 分
2. **中期爬坡**：实体完整性和纹理质感在 30–120 万步持续爬升，驱动总分从 71 分提升至 79 分
3. **后期觉醒**：计数能力在 55 万步时突然爆发，标志着高阶能力的冲刺窗口

在效率维度上，本模型仅使用 **300 GPU 小时**（单卡 A100，云端租金约 2,500 元），取得了 **79.57 分**的 DPG-Bench 成绩，得分效率是 SeFi-Image 的数百倍，并已超越 2.6B 参数的 SDXL。

---

## 主要结果

### DPG-Bench 成长曲线（30万–120万步）

| 训练步数 | Relation | Entity | Attribute | Global | Other | **Overall** |
|---------|----------|--------|-----------|--------|-------|------------|
| 30万 | 92.69 | 82.13 | 84.24 | 79.14 | 57.20 | **71.09** |
| 50万 | 92.54 | 84.83 | 87.02 | 82.21 | 62.00 | **75.27** |
| 75万 | 93.63 | 86.53 | 87.77 | 80.67 | 74.40 | **77.85** |
| 100万 | 94.17 | 87.73 | 88.45 | 79.75 | 70.40 | **79.57** |
| 120万 | 94.48 | 87.29 | 88.23 | 81.29 | 70.40 | **79.00** |

完整的训练日志、路由行为分析及 DPG-Bench 评测数据详见 [`logs/`](#文件结构) 目录。

![DPG-Bench L1 指标成长曲线](logs/dpg_bench_L1.png)

*图：DPG-Bench 五项 L1 指标及总分随训练步数的变化（30万–120万步）。空间关系（Relation）早期定型，实体（Entity）与纹理（Attribute）持续爬坡，计数（Other）后期觉醒。*


<img src="assert/ScreenShot_2026-07-30_001153_698.png" width="100%"/>
*125 万步 checkpoint 的 DPG-Bench 采样图展示*


### 与业界模型对比

| 模型 | 参数量 | 训练成本 | Overall |
|------|--------|---------|---------|
| SD v1.5 | 0.86B | — | 63.18 |
| SDXL | 2.6B | — | 74.65 |
| **LingXi-Image-MoE (1.68B)** | **1.68B** | **300 GPU时** | **79.57** |
| PixArt-Σ | 0.6B | — | 80.54 |
| FLUX.1 Dev | 12B | — | 83.84 |

### 训练成本与数据效率

| 模型 | GPU 小时 | 数据量 | 得分 | 效率(分/GPU时) |
|------|---------|--------|------|---------------|
| Z-Image | 314,000 H800 | 未公开 | 88.14 | 0.00028 |
| SeFi-Image 5B | 125,000 A800 | 4.5亿 | 87.27 | 0.00070 |
| **LingXi-Image-MoE** | **300 A100** | **266万** | **79.57** | **0.265** |

> 得分效率是 Z-Image 的 **940 倍以上**，是 SeFi-Image 的 **380 倍以上**。

---

## 跨分辨率泛化

2D RoPE 连续位置编码使得单一权重可在任意分辨率下推理。以下为 125 万步 checkpoint 的 256 基座、512 零样本与 150 万步 512 微调的生成对比：

| 256×256（基座，125万步） | 512×512（零样本，125万步） | 512×512（微调，150万步） |
|:---:|:---:|:---:|
| <img src="assert/step1250000/img256_cfg4.0_seed0_t2i/0000_An anthropomorphic rainbow fox_ its fur dotted wit.png" width="100%"/> | <img src="assert/step1250000/img512_cfg4.0_seed0_t2i/0000_An anthropomorphic rainbow fox_ its fur dotted wit.png" width="100%"/> | <img src="assert/step1500000/img512_cfg4.0_seed0_t2i/0000_An anthropomorphic rainbow fox_ its fur dotted wit.png" width="100%"/> |
| <img src="assert/step1250000/img256_cfg4.0_seed0_t2i/0000_a spectacular display of fireworks illuminates the.png" width="100%"/> | <img src="assert/step1250000/img512_cfg4.0_seed0_t2i/0000_a spectacular display of fireworks illuminates the.png" width="100%"/> | <img src="assert/step1500000/img512_cfg4.0_seed0_t2i/0000_a spectacular display of fireworks illuminates the.png" width="100%"/> |
| <img src="assert/step1250000/img256_cfg4.0_seed0_t2i/0000_A fantasy dragon_ its body is dark purple gradient.png" width="100%"/> | <img src="assert/step1250000/img512_cfg4.0_seed0_t2i/0000_A fantasy dragon_ its body is dark purple gradient.png" width="100%"/> | <img src="assert/step1500000/img512_cfg4.0_seed0_t2i/0000_A fantasy dragon_ its body is dark purple gradient.png" width="100%"/> |
| <img src="assert/step1250000/img256_cfg4.0_seed0_t2i/0000_A bustling vintage bookstore interior bathed in wa.png" width="100%"/> | <img src="assert/step1250000/img512_cfg4.0_seed0_t2i/0000_A bustling vintage bookstore interior bathed in wa.png" width="100%"/> | <img src="assert/step1500000/img512_cfg4.0_seed0_t2i/0000_A bustling vintage bookstore interior bathed in wa.png" width="100%"/> |

以下为 DPG-Bench 定量对比：

### 零样本与 512 微调后的 DPG-Bench 分数对比

| 训练配置 | Global | Entity | Attribute | Relation | Other | **Overall** |
|---------|--------|--------|-----------|----------|-------|------------|
| 256（125万步，基座） | 80.67 | 87.40 | 88.25 | 94.24 | 70.39 | **79.07** |
| 256→512（零样本） | 68.40 | 70.44 | 71.59 | 85.96 | 39.20 | **55.63** |
| 512微调 5万步 | 76.68 | 87.10 | 86.94 | 93.98 | 71.60 | **77.74** |
| 512微调 10万步 | 77.91 | 87.35 | 87.06 | 93.82 | 74.40 | **78.53** |
| 512微调 15万步 | 77.91 | 87.29 | 87.06 | 93.16 | 71.20 | **78.02** |
| 512微调 20万步 | 77.91 | 87.51 | 87.16 | 93.43 | 74.80 | **78.53** |
| 512微调 25万步 | 77.61 | 87.24 | 87.63 | 93.74 | 77.20 | **78.25** |

**关键结论：**

1. **全层 MoE 架构具备良好的跨分辨率泛化潜力。** 零样本下 Relation 仍保持 85.96 分，证明 2D RoPE 有效迁移了宏观空间推理能力。
2. **高频细节随分辨率下降。** Entity/Attribute 从 87–88 分降至 70–71 分，体现为零样本推理中的"水波纹"伪影。
3. **短时微调即可快速恢复。** 512 微调仅 5 万步后总分即恢复至 77.74 分（基座 79.07分）
---

## 核心发现

### 能力涌现三阶段假说

1. **阶段 I：早期定型（0–30万步）**——空间关系能力迅速收敛到 92+ 分，此后几乎不再变化
2. **阶段 II：中期爬坡（30–80万步）**——实体完整性和纹理质感成为增长主力，驱动总分持续提升
3. **阶段 III：后期攻坚（80万步+）**——计数等高阶能力在基础能力饱和后迎来爆发

### Global 指标的"伪饱和"与文本编码器瓶颈

Global 指标在 79–82 分之间长期高位震荡。结合 0.47B 模型同样使用 T5-base 的对比，推断这一瓶颈源于 **轻量级文本编码器（T5-base，220M）的语义表达上限**，而非全层 MoE 架构的生成能力不足。升级文本编码器（如 T5-XXL 或 Qwen-VL）将是突破关键。

### U 型分层专业化

浅层（Layer 0–2）和深层（Layer 9–11）的路由分化度（CV）显著高于中间层（Layer 3–8），这一模式在 0.47B 和 1.68B 模型上均得到复现，是视觉扩散 MoE 的结构性特征。

---

## 核心特性

### ProMoE 两阶段路由
- **条件路由**：按功能角色将 token 分为有条件（CFG 中未被 dropout 的样本）和无条件（CFG 空文本样本）两组
- **原型路由**：通过可学习的原型向量与 token 的余弦相似度进行路由分配，每个 token 激活 Top-K 个路由专家
- **无条件专家重定位**：无条件专家（Expert 12）专门处理 CFG 空文本样本

### MM-DiT 文本-图像融合
- 文本序列与图像 token 沿序列维度拼接，自注意力同时处理文本和图像
- 文本全局向量通过 AdaLN 调制注入模型

### 动态分辨率（2D 连续旋转位置编码）
- 图像 token：2D RoPE，坐标归一化到 [0,1] 区间
- 文本 token：1D RoPE
- 单一权重支持任意分辨率（256/512/768/1024+），无需插值或微调
- 已验证 256→512 零样本跨分辨率泛化能力（见[跨分辨率泛化](#跨分辨率泛化)）

### 统一共生框架
- 文生图（T2I）：源潜变量为零黑图
- 图像编辑（Image Editing）：源潜变量为目标图像的原始编码

---

## 模型架构

### ProMoE-TC-L 配置（1.68B 参数）

| 参数 | 值 |
|------|------|
| 隐藏层维度 | 1024 |
| Transformer 层数 | 24（全部 MoE） |
| 注意力头数 | 16 |
| 路由专家数 | 12/层 |
| 无条件专家数 | 1/层 |
| 共享专家数 | 1/层 |
| Top-K 激活 | 2 |
| 中间层维度（MoE） | 2048 |
| 总参数量 | **1.67B** |
| 激活参数量（每 token） | **559.7M** |
| 模型大小（float32） | 6,362 MB |

**参数量分解（total / active per token）：**

| 组件 | 总参数量 | 激活参数量 |
|------|---------|-----------|
| Attention | 251.9M | 251.9M |
| MoE 专家 | 1,309.6M | 201.5M |
| 共享专家 | 100.7M | 100.7M |
| Embedding | 3.2M | 3.2M |
| Router | 0.3M | 0.3M |
| Final | 2.1M | 2.1M |

**激活内存估算（batch=1, float32）：** 256px: 2.4GB | 512px: 4.8GB | 1024px: 14.3GB

> 所有 24 层均使用 MoE（代码中 `use_moe_flag = [True] * depth` 覆盖 interleave 配置）。

---

## 环境配置

```bash
conda create -n promoe python=3.10 -y
conda activate promoe
pip install -r requirements.txt
```

---

## 数据预处理

使用 T5-base 编码文本、sd-vae-ft-mse 编码图像，预处理为缓存文件以加速训练。

### 1. 下载数据

```bash
bash preprocess/download.sh
```

并发下载 `train-000000.tar` ~ `train-001310.tar`，默认保存到 `/data/coding/`。

### 2. 解压数据

```bash
python preprocess/extracted_data.py
```

解压后目录结构：
```
curated_extracted_data/
└── synthetic_original_prompt_square_resolution/
    ├── 000000000.jpg / 000000000.txt
    ├── 000000001.jpg / 000000001.txt
    └── ...
```

### 3. 提取特征

```bash
python preprocess/preprocess_ma-xu-fine-t2i.py \
  --image_root /path/to/curated_extracted_data \
  --output_dir /path/to/preprocessed_output \
  --t5_path /path/to/t5-base \
  --vae_path /path/to/sd-vae-ft-mse \
  --image_size 512 \
  --batch_size 8
```

输出结构：
```
preprocessed_output/
├── latents/
│   ├── target_512/            # 512 分辨率潜变量
│   └── target/                # 256 分辨率潜变量
└── t5_text_features/
    ├── 000000000_seq.npy      # T5 序列特征 (512, 768)
    ├── 000000000_mask.npy     # 注意力掩码 (512,)
    └── ...
```

### 4. 配置训练

```yaml
latent_root: "/path/to/preprocessed_output/latents"
text_root: "/path/to/preprocessed_output/t5_text_features"
```

---

## 数据集构成

训练使用约 **266 万** 图文对，按约 **8.5:1.5** 比例混合文生图与图像编辑数据：

| 子集 | 规模 | 说明 |
|------|------|------|
| Fine-T2I enhanced | ~154 万 | 长文本复杂细节 |
| Fine-T2I original | ~44 万 | 短文本泛化（从 130 万中采样） |
| Fine-T2I curated | ~16.8 万 | 真实摄影，消除"AI 塑料感" |
| 人脸/文字增强 | ~19.2 万 | 提升面部与文字渲染 |
| 图像编辑数据 | ~28 万 | 开源编辑数据集 |
| **总计** | **~266 万** | |

---

## 训练

### 两步联合训练

使用 `train_joint.sh` 脚本，默认配置文件为 `configs/joint.yaml`（256 分辨率）。

**第一步：基础训练（20 万步）**

```bash
bash train_joint.sh
```

**第二步：从 checkpoint 恢复继续训练（200 万步）**

```bash
bash train_joint.sh --ckpt ProMoE_L/ProMoE_TC_L/joint_d4HRU/checkpoints/ckpt_step_1250000.pth --steps 2000000
```

**512 分辨率训练：**

```bash
bash train_joint.sh --size 512 --ckpt path/to/checkpoint.pth --steps 2000000
```

### 自定义参数

```bash
# 分段训练（学习率/步数可变）
bash train_joint.sh --lr 1e-4 5e-5 --steps 150000 50000

# 完整参数
bash train_joint.sh --ckpt path --accum 4 --batch_size 32 --img_size 512 --gpu 0
```

### 训练配置（configs/joint.yaml 与 configs/joint_512.yaml）

| 参数 | 256 分辨率 | 512 分辨率 |
|------|-----------|-----------|
| `image_size` | 256 | 512 |
| `total_train_batch_size` | 64 | 16 |
| `lr` | 1e-4 | 1e-4 |
| `latent_root` | `.../latents` | `.../latents_512` |
| 配置文件名 | `joint.yaml` | `joint_512.yaml` |

### 训练成本

- 硬件：单卡 A100 80GB
- 时间：约 12 天（至 120 万步）
- 成本：约 2,500 元人民币（云端租金）
- 所有 DPG-Bench 评测：NVIDIA RTX 4080 本地完成

> 从 0.47B 模型中系统性总结的 Loss Spike 修复方案（CosineAnnealingLR、VAE 均值处理、数据完整性校验、优化器状态重置）已全部迁移至本模型，整个训练过程 Loss 从 ~1.6 平滑下降至 0.4–0.5，未出现任何 Spike。

---

## 推理

本仓库提供三种推理方式。`inference.py` 和 `sample.py` 直接加载 `.pth` 权重进行推理。`infer_diffuser.py` 使用 Diffusers 格式，需先用 `pt2diffuser.py` 将 checkpoint 转换：

```bash
python pt2diffuser.py --config configs/joint.yaml \
    --ckpt path/to/ckpt_step_1250000.pth \
    --save_dir ./pro_moe_diffusers \
    --vae_path /path/to/sd-vae-ft-mse \
    --t5_path /path/to/t5-base \
    --image_size 256
```

| 参数 | 说明 |
|------|------|
| `--config` | YAML 配置文件（指定模型架构） |
| `--ckpt` | 训练好的 checkpoint `.pth` 文件 |
| `--save_dir` | 转换后 Diffusers 格式的输出目录 |
| `--vae_path` | VAE 模型路径 |
| `--t5_path` | T5 文本编码器路径 |
| `--image_size` | 训练图像尺寸（默认 256） |

### 1. inference.py（轻量推理）

```bash
# 单 prompt
python inference.py \
  --config configs/joint.yaml \
  --ckpt path/to/checkpoint.pth \
  --prompt "A cat holds a poster" \
  --output outputs \
  --vae_path /path/to/sd-vae-ft-mse \
  --t5_path /path/to/t5-base \
  --guide_scale 4.0 \
  --seed 42

# 批量推理
python inference.py \
  --config configs/joint.yaml \
  --ckpt path/to/checkpoint.pth \
  --prompt_dir /path/to/prompt/dir \
  --output outputs
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | 必填 | YAML 配置文件 |
| `--ckpt` | 必填 | 模型 checkpoint 路径 |
| `--prompt` | — | 单条文本 prompt |
| `--prompt_dir` | — | 存放 .txt 文件的目录 |
| `--output` | outputs | 输出目录 |
| `--image_size` | 256 | 生成图像尺寸 |
| `--guide_scale` | 4.0 | CFG 引导强度 |
| `--sample_steps` | 50 | 采样步数 |
| `--seed` | 0 | 随机种子 |
| `--vae_path` | — | VAE 模型路径 |
| `--t5_path` | — | T5 模型路径 |

### 2. sample.py（采样与评估）

支持多 checkpoint 对比、多 CFG 尺度、图像编辑任务：

```bash
python sample.py \
  --config configs/joint.yaml \
  --step_list_for_sample 1000000,1200000 \
  --guide_scale_list 4.0,5.0,7.0 \
  --sample_prompts "a cat sitting on a chair" \
  --source_image_path /path/to/source.jpg
```

| 参数 | 说明 |
|------|------|
| `--step_list_for_sample` | checkpoint 步数列表（逗号分隔） |
| `--guide_scale_list` | CFG 引导尺度列表 |
| `--num_fid_samples` | 每步采样数量 |
| `--source_image_path` | 源图像路径（编辑任务） |
| `--sample_prompts` | 直接指定 prompt |
| `--prompt_file` | prompt 文件（每行一条） |

### 3. infer_diffuser.py（Diffusers 格式推理）

从 diffusers 格式加载转换后的模型进行推理：

```bash
# 单 prompt
python infer_diffuser.py \
  --model_dir ./LingXi-Image-MoE-1.68B-A0.56M \
  --prompt "A cat" \
  --output ./outputs

# 批量推理
python infer_diffuser.py \
  --model_dir ./LingXi-Image-MoE-1.68B-A0.56M \
  --prompt_dir ./prompts \
  --output ./outputs \
  --guide_scale 4.0 \
  --steps 50

# 仅分析模型参数量/激活量
python infer_diffuser.py --model_dir ./pro_moe_diffusers --analyze
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_dir` | 必填 | 转换后的 diffusers 模型目录 |
| `--prompt` | — | 单条文本 prompt |
| `--prompt_dir` | — | 批量 prompt 目录 |
| `--output` | outputs | 输出目录 |
| `--guide_scale` | 4.0 | CFG 引导强度 |
| `--steps` | 50 | 采样步数 |
| `--seed` | 0 | 随机种子 |
| `--analyze` | — | 仅分析模型参数量 |

---

## 训练稳定性

从 0.47B 模型中系统性总结的 Loss Spike 修复方案已全部迁移：

| 问题 | 根本原因 | 修复方案 |
|------|---------|---------|
| 后期震荡 | 固定 lr | CosineAnnealingLR |
| 数值噪声 | VAE 方差采样 | 直接取均值 |
| 梯度爆炸 | 损坏文件含 NaN/Inf | Dataset 层过滤 |
| 恢复后崩溃 | 动量被异常梯度污染 | 重置优化器状态 |
| 偶发 OOM | Loss 阈值过低 | 提高阈值至 1.5 |

整个 1.68B 训练过程 Loss 平滑下降，**未出现任何 Loss Spike**。

---

## 已知局限

- **人脸细节**：受限于 4 通道 VAE 和 T5-base 容量，偶尔出现崩坏
- **计数能力**：DPG-Bench count 约 68.5 分，仍需专项数据攻坚
- **Global 指标**：受 T5-base 语义表达上限限制，79–82 分之间震荡
- **512 分辨率**：零样本推理存在高频伪影，需微调修复
- **显存需求**：1.68B 全层 MoE 在4080显卡上可以运行

---

## 后续工作

1. **升级文本编码器**：T5-XXL 或 Qwen-VL，突破 Global 瓶颈
2. **大规模数据扩充**：补充计数专项数据
3. **分阶段课程学习**：256→512 高分辨率训练
4. **升级 VAE**：16 通道 VAE 突破细节上限

预期总训练步数 150–200 万步（含高分辨率阶段）时，总分有望冲击 **85 分以上**，总 GPU 消耗预计控制在 2,000 GPU 小时以内。

---

## 文件结构

```
├── configs/
│   ├── joint.yaml               # 联合训练配置 (256)
│   ├── joint_512.yaml           # 联合训练配置 (512)
│   ├── 004_ProMoE_L.yaml        # ProMoE-L 参考配置
│   └── ...
├── train_joint.sh               # 联合训练流水线脚本
├── train_GradientAccumulationSteps.py  # 训练脚本
├── models/
│   ├── modules.py               # 共享模块 (Attention, RoPE, MoE)
│   └── models_ProMoE_TC.py     # ProMoE 文本条件模型
├── preprocess/                  # 数据预处理脚本
├── inference.py                 # 轻量推理脚本
├── sample.py                    # 采样与评估脚本
├── infer_diffuser.py            # Diffusers 格式推理脚本
├── pt2diffuser.py               # 模型格式转换工具
├── logs/                        # 训练日志与可视化
│   ├── joint_phase1.log         # 联合训练完整日志（0–145万步）
│   ├── moe_routing.log          # 路由专家利用率日志（每100步）
│   ├── dpg-bench.txt            # DPG-Bench 完整评测记录
│   ├── dpg_bench_L1.png         # L1 指标成长曲线图
│   ├── dpg_bench_L2.png         # L2 子指标成长曲线图
│   ├── loss_comparison.png      # 训练 Loss 曲线图
│   ├── plot_dpg_bench.py        # DPG-Bench 可视化脚本
│   ├── plot_loss.py             # Loss 曲线绘制脚本
│   └── analyze_routing.py       # 路由行为分析脚本
└── evaluation/                  # 评估脚本
```

---

## 引用

```bibtex
@misc{sha2026fullayer,
  title={From Alternating to Full-Layer: A Comparative Study of Mixture-of-Experts Architectures for Text-to-Image Diffusion Models},
  author={Haiying Sha},
  year={2026},
  note={LingXi-Image-MoE-1.0},
  url={https://github.com/Lingxi-Qihang/LingXi-Image-MoE}
}

@article{sha2026emergence,
  title={Temporal Patterns of Capability Emergence: A Fine-Grained Analysis of Training Dynamics in a 1.68B Full-Layer MoE Text-to-Image Mode},
  author={Haiying Sha},
  year={2026},
  note={LingXi-Image-MoE-1.68B},
  url={https://github.com/Lingxi-Qihang/LingXi-Image-MoE-1.68B-A0.56M}
}
```

---

## 参考文献

[1] DiT: Scalable Diffusion Models with Transformers. ICCV 2023.  
[2] ProMoE: Routing Matters in MoE. ICLR 2026.  
[3] FiT: Flexible Vision Transformer for Diffusion Model. NeurIPS 2024.  
[4] DPG-Bench: A Dense Prompt Generation Benchmark for Text-to-Image Models. ICLR 2024.  
[5] Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. ICML 2024.  
[6] Sparse MoE Routing in Visual Diffusion Transformers: Diagnosis and Roadmap. arXiv:2605.19378.
