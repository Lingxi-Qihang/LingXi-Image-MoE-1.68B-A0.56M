#!/usr/bin/env python3
"""
pt2diffuser.py — 将 ProMoE checkpoint 转换为 diffusers 格式

用法：
  python pt2diffuser.py --config configs/joint.yaml \
      --ckpt ProMoE_L/ProMoE_TC_L/joint/checkpoints/ckpt_step_750000.pth \
      --save_dir ./pro_moe_diffusers \
      --vae_path E:/AImodel/ProMoE/sd-vae-ft-mse \
      --t5_path E:/AImodel/t5-base
"""
import os, sys, json, argparse, shutil
import torch
import yaml

# 将上级目录加入 path，让模型能找到 models/ 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer

from config import cfg as base_cfg
from train_GradientAccumulationSteps import model_dict
from promoe_pipeline import ProMoEDiTModel, ProMoEPipeline


# ==================== 转换主函数 ====================

def convert_checkpoint(args):
    print("=" * 60)
    print("ProMoE → Diffusers 格式转换")
    print("=" * 60)

    device = torch.device("cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 读取 YAML 配置
    with open(args.config, encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f)

    model_name = yaml_cfg["model_name"]
    model_class, config_name = model_dict[model_name]

    # 合并配置
    model_cfg = dict(getattr(base_cfg, config_name, {}))
    model_cfg.update(yaml_cfg.get(config_name, {}))
    model_cfg["input_size"] = args.image_size // 8

    print(f"模型: {model_name}")
    print(f"配置: hidden_size={model_cfg.get('hidden_size')}, depth={model_cfg.get('depth')}")

    # 2. 加载权重
    print(f"加载 checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt.get("ema_model_state_dict", ckpt.get("model_state_dict", ckpt))

    # 3. 构建模型并加载权重（model_name 不传给模型构造函数）
    print("构建模型...")
    model = model_class(**model_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}")
    model.eval()

    # 4. 包装为 diffusers 模型（需要 model_name 来查找模型类）
    dit_model = ProMoEDiTModel({**model_cfg, "model_name": model_name})
    dit_model.dit.load_state_dict(model.state_dict())
    dit_model = dit_model.to("cpu")

    # 5. 保存 DiT 模型
    dit_save_dir = os.path.join(args.save_dir, "dit_model")
    os.makedirs(dit_save_dir, exist_ok=True)
    dit_model._save_to_safetensors(dit_save_dir)

    # 保存 DiT 配置
    dit_config = {**model_cfg, "model_name": model_name}
    with open(os.path.join(dit_save_dir, ProMoEDiTModel.config_name), "w") as f:
        json.dump(dit_config, f, indent=2, default=str)

    # 6. 保存 VAE
    print("保存 VAE...")
    vae = AutoencoderKL.from_pretrained(args.vae_path)
    vae.save_pretrained(os.path.join(args.save_dir, "vae"))

    # 7. 保存 T5 文本编码器
    print("保存 T5...")
    tokenizer = T5Tokenizer.from_pretrained(args.t5_path)
    text_encoder = T5EncoderModel.from_pretrained(args.t5_path)
    tokenizer.save_pretrained(os.path.join(args.save_dir, "text_encoder"))
    text_encoder.save_pretrained(os.path.join(args.save_dir, "text_encoder"))

    # 8. 保存 scheduler 配置
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=1.0,
    )
    scheduler.save_pretrained(os.path.join(args.save_dir, "scheduler"))

    # 9. 保存 pipeline
    print("保存 pipeline 配置...")
    pipeline = ProMoEPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_model=dit_model,
        scheduler=scheduler,
    )
    pipeline.save_pretrained(args.save_dir)

    print(f"转换完成! 模型已保存至: {args.save_dir}")
    print(f"推理命令:")
    print(f"  python infer_diffuser.py --model_dir {args.save_dir} \\")
    print(f'      --prompt "A cat" --output ./outputs')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProMoE → Diffusers 格式转换")
    parser.add_argument("--config", default="configs/joint.yaml", help="YAML config (e.g. configs/joint.yaml)")
    parser.add_argument("--ckpt", default=r"D:\mywork\pythonProject\LingXi-Image-MoE - 副本\ProMoE_L\ProMoE_TC_L\joint\checkpoints\ckpt_step_1250000.pth", help="checkpoint .pth 路径")
    parser.add_argument("--save_dir", default="LingXi-Image-MoE-1.68B-A0.45M", help="输出目录")
    parser.add_argument("--vae_path", default="E:/AImodel/ProMoE/sd-vae-ft-mse", help="VAE 路径")
    parser.add_argument("--t5_path", default="E:/AImodel/t5-base", help="T5 模型路径")
    parser.add_argument("--image_size", type=int, default=256, help="训练图像尺寸")
    args = parser.parse_args()
    convert_checkpoint(args)
