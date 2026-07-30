#!/usr/bin/env python3
"""
infer_diffuser.py — 从 diffusers 格式加载 ProMoE 并推理

用法：
  python infer_diffuser.py --model_dir ./pro_moe_diffusers \
      --prompt "A cat" --output ./outputs

  python infer_diffuser.py --model_dir ./pro_moe_diffusers \
      --prompt_dir ./prompts --output ./outputs --guide_scale 4.0 --steps 50
"""
import os, sys, argparse, glob, logging
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# 将脚本所在目录加入 path（加载 pt2diffuser.py 中的 ProMoEPipeline）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s-%(levelname)s]: %(message)s")


def load_prompts(prompt, prompt_dir):
    if prompt_dir:
        txt_files = sorted(glob.glob(os.path.join(prompt_dir, "*.txt")))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {prompt_dir}")
        prompts = []
        for f in txt_files:
            with open(f, "r", encoding="utf-8") as fh:
                content = fh.read().strip()
            name = os.path.splitext(os.path.basename(f))[0]
            prompts.append((name, content))
        logging.info(f"Loaded {len(prompts)} prompts from {prompt_dir}")
        return prompts
    return [("prompt", prompt)]


# ==================== 模型参数量 / 激活量分析 ====================

def count_params(module: nn.Module) -> int:
    """计算模块的总参数量"""
    return sum(p.numel() for p in module.parameters())


def analyze_model_stats(model) -> dict:
    """
    分析 ProMoE 模型的参数量和激活量。
    返回格式类似 "30B-A3B" 即 "总参-激活参"。
    """
    dit = model.dit if hasattr(model, "dit") else model
    total_params = count_params(dit)
    total_bytes = total_params * 4  # float32

    info = {
        "total_params": total_params,
        "total_params_str": f"{total_params/1e9:.2f}B",
        "total_size_mb": total_bytes / (1024 ** 2),
    }

    # 逐层分析
    per_layer = {"embed": 0, "attn": 0, "dense_mlp": 0, "moe": 0, "shared_expert": 0, "router": 0, "final": 0}
    active_per_layer = {"embed": 0, "attn": 0, "dense_mlp": 0, "moe_active": 0, "shared_expert": 0, "router": 0, "final": 0}

    # 1. Embedding 层（始终激活）
    for name in ["x_embedder", "t_embedder", "text_proj"]:
        p = count_params(getattr(dit, name, nn.Identity()))
        per_layer["embed"] += p
        active_per_layer["embed"] += p

    # 2. Final 层
    per_layer["final"] = count_params(dit.final_layer)
    active_per_layer["final"] = per_layer["final"]

    # 3. null_text_embed
    # 它是可学习的，计入 total 但不影响激活量（只有 1*768 很小）
    null_p = dit.null_text_embed.numel()
    per_layer["embed"] += null_p
    active_per_layer["embed"] += null_p

    # 4. 逐 block 分析
    top_k = dit.MoE_config.get("top_k", 2) if dit.MoE_config else 2
    moe_config = dit.MoE_config or {}
    use_shared = moe_config.get("use_shared_expert", True)
    use_interleave = moe_config.get("interleave", False)
    num_routed = moe_config.get("num_routed_experts", 12)
    use_uncond = moe_config.get("use_uncond_expert", True)
    num_experts = num_routed + (1 if use_uncond else 0)

    for i, block in enumerate(dit.blocks):
        is_moe = block.use_moe

        # 2a. Attention（始终激活）
        attn_p = count_params(block.attn)
        norm_p = count_params(block.norm1) + count_params(block.norm2)
        adaln_p = count_params(block.adaLN_modulation)
        per_layer["attn"] += attn_p + norm_p + adaln_p
        active_per_layer["attn"] += attn_p + norm_p + adaln_p

        if is_moe:
            # MoE block
            moe = block.mlp
            # Router cluster centers (nn.Parameter, 直接 numel)
            router_p = moe.cluster_centers.numel()
            per_layer["router"] += router_p
            active_per_layer["router"] += router_p

            # 所有 expert 权重（计入 total）
            for expert in moe.experts:
                per_layer["moe"] += count_params(expert)

            # Active MoE: top_k 个 routed experts + unconditional expert（如果存在则专门处理）
            # unconditional expert 只在无条件 token 时激活，这里按有条件的 case 算
            expert_active_p = 0
            for expert in moe.experts[:num_routed]:
                expert_active_p += count_params(expert)
            # 实际激活：top_k 个 routed experts
            active_per_layer["moe_active"] += expert_active_p * (top_k / num_routed)

            # Shared expert（始终激活）
            if use_shared and hasattr(moe, "shared_expert") and moe.shared_expert is not None:
                shared_p = count_params(moe.shared_expert)
                per_layer["shared_expert"] += shared_p
                active_per_layer["shared_expert"] += shared_p
        else:
            # Dense MLP
            dense_p = count_params(block.mlp)
            per_layer["dense_mlp"] += dense_p
            active_per_layer["dense_mlp"] += dense_p

    # 统计
    total = sum(per_layer.values())
    total_active = sum(active_per_layer.values())

    info["per_layer"] = {k: f"{v/1e6:.1f}M" for k, v in per_layer.items()}
    info["active_per_layer"] = {k: f"{v/1e6:.1f}M" for k, v in active_per_layer.items()}
    info["active_params"] = total_active
    info["active_params_str"] = f"{total_active/1e6:.1f}M"
    info["summary"] = f"{total/1e6:.0f}M-{total_active/1e6:.0f}M"

    # 激活内存估算（forward 时的中间激活）
    # 对于 DiT，主要激活 = seq_len * hidden_size * depth * bytes * 几个副本
    hidden_size = dit.hidden_size
    depth = len(dit.blocks)
    patch_size = dit.patch_size
    # 假设 256x256 图像: latent=32x32, tokens = (32/2)^2 = 256 + 512(text) = 768
    # 或者用传入的 input_size 估算
    for img_size in [256, 512, 1024]:
        latent_h = img_size // 8
        num_patches = (latent_h // patch_size) ** 2
        seq_len = num_patches + 512  # + text tokens
        # 每层的主要激活: attention scores (B, H, N, N) + hidden states
        # 近似: seq_len * hidden_size * 34 字节（qkv, attn_out, mlp_hidden, residual 等）
        act_per_token = hidden_size * 34 * 4  # bytes
        act_per_layer_bytes = seq_len * act_per_token
        total_activation = act_per_layer_bytes * depth
        act_gb = total_activation / (1024 ** 3)
        info.setdefault("activation_estimation", {})[f"{img_size}px"] = f"{act_gb:.1f}GB"

    info["config"] = {
        "hidden_size": dit.hidden_size,
        "depth": depth,
        "num_heads": dit.num_heads,
        "num_routed_experts": num_routed,
        "top_k": top_k,
        "interleave": use_interleave,
        "use_shared_expert": use_shared,
    }

    return info


def print_model_stats(info: dict):
    """打印模型统计信息"""
    print("=" * 60)
    print("  ProMoE 模型参数量分析")
    print("=" * 60)
    print(f"  总参数量:       {info['total_params_str']} ({info['total_params']:,})")
    print(f"  激活参数量:     {info['active_params_str']} (每 token)")
    print(f"  总参-激活参:    {info['summary']}")
    print(f"  模型大小:       {info['total_size_mb']:.0f} MB (float32)")
    print()
    print("  配置:")
    for k, v in info["config"].items():
        print(f"    {k}: {v}")
    print()
    print("  参数量分解 (total / active):")
    all_keys = set(list(info["per_layer"].keys()) + list(info["active_per_layer"].keys()))
    for k in sorted(all_keys):
        t = info["per_layer"].get(k, "0M")
        a = info["active_per_layer"].get(k, "0M")
        print(f"    {k:20s}  total={t:>8s}  active={a:>8s}")
    print()
    print("  激活内存估算 (forward, batch=1, float32):")
    for res, gb in info["activation_estimation"].items():
        print(f"    {res:>6s}: {gb}")
    print("=" * 60)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="ProMoE Diffusers 推理")
    parser.add_argument("--model_dir", required=True, help="转换后的模型目录")
    parser.add_argument("--prompt", type=str, default=None, help="单条文本提示")
    parser.add_argument("--prompt_dir", type=str, default=None, help="批量 prompt 目录")
    parser.add_argument("--output", type=str, default="outputs", help="输出目录")
    parser.add_argument("--guide_scale", type=float, default=4.0, help="CFG 引导尺度")
    parser.add_argument("--steps", type=int, default=50, help="采样步数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--analyze", action="store_true", help="仅分析模型参数量/激活量，不推理")
    args = parser.parse_args()

    if not args.prompt and not args.prompt_dir and not args.analyze:
        parser.error("需要 --prompt 或 --prompt_dir（或使用 --analyze）")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- 加载 pipeline ----
    logging.info("加载 pipeline...")
    from promoe_pipeline import ProMoEPipeline
    model_dir = args.model_dir
    pipeline = ProMoEPipeline.from_pretrained(model_dir)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    logging.info("加载完成")

    # ---- 分析模式 ----
    if args.analyze:
        info = analyze_model_stats(pipeline.dit_model)
        print_model_stats(info)
        return

    # ---- 加载 prompts ----
    prompts = load_prompts(args.prompt, args.prompt_dir)
    os.makedirs(args.output, exist_ok=True)

    # ---- 推理 ----
    generator = torch.Generator(device=device) if args.seed >= 0 else None

    for idx, (name, prompt_text) in enumerate(tqdm(prompts, desc="Prompts")):
        if generator is not None:
            generator.manual_seed(args.seed + idx)

        images = pipeline(
            prompt=prompt_text,
            generator=generator,
            num_inference_steps=args.steps,
            guidance_scale=args.guide_scale,
        )

        if isinstance(images, list):
            for j, img in enumerate(images):
                suffix = f"_{j}" if len(images) > 1 else ""
                img.save(os.path.join(args.output, f"{name}{suffix}.png"))
        elif isinstance(images, torch.Tensor):
            torchvision.utils.save_image(
                images.float() / 255.0,
                os.path.join(args.output, f"{name}.png"),
            )

    logging.info(f"全部完成，结果保存在 {args.output}")


if __name__ == "__main__":
    main()
