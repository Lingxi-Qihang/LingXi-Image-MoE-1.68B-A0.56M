import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.models import AutoencoderKL
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def extract_vae_latent(image_path, vae, transform, device):
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        latent = vae.encode(img_tensor).latent_dist.parameters
        latent_flip = vae.encode(img_tensor.flip(dims=[3])).latent_dist.parameters
        return latent.squeeze(0).detach().cpu().numpy(), latent_flip.squeeze(0).detach().cpu().numpy()
    except Exception:
        return None, None


def extract_t5_features(prompts, tokenizer, text_encoder, device, max_length=512):
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    hidden_states = outputs.last_hidden_state.cpu().float().numpy()
    masks = inputs.attention_mask.cpu().bool().numpy()
    return hidden_states, masks


def main(image_root, jsonl_path, output_dir, t5_path, vae_path, image_size=256, batch_size=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载 T5
    print("Loading T5...")
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    text_encoder = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.bfloat16).eval().to(device)

    # 加载 VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(vae_path).eval().to(device)

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 读取 JSONL
    print(f"Reading JSONL from {jsonl_path}...")
    all_samples = []  # 存储每个样本的完整信息（包括路径和输出路径）
    skipped_empty = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: line {line_num} JSON decode error, skipping.")
                continue

            src_path = item.get('local_input_image')
            tgt_path = item.get('output_image')
            prompt = item.get('text', '')

            if not prompt:
                skipped_empty += 1
                continue
            if not src_path or not tgt_path:
                print(f"Warning: line {line_num} missing local_input_image or output_image, skipping.")
                continue

            src_full = os.path.normpath(os.path.join(image_root, src_path))
            tgt_full = os.path.normpath(os.path.join(image_root, tgt_path))

            # 生成输出文件路径
            tgt_base = os.path.splitext(os.path.basename(tgt_full))[0]
            src_base = os.path.splitext(os.path.basename(src_full))[0]

            t5_seq = os.path.join(output_dir, 't5_text_features', f"{tgt_base}_seq.npy")
            t5_mask = os.path.join(output_dir, 't5_text_features', f"{tgt_base}_mask.npy")
            vae_src = os.path.join(output_dir, 'latents', 'source', f"{src_base}.latent.npz")
            vae_tgt = os.path.join(output_dir, 'latents', 'target', f"{tgt_base}.latent.npz")

            sample = {
                'src': src_full,
                'tgt': tgt_full,
                'prompt': prompt,
                't5_seq': t5_seq,
                't5_mask': t5_mask,
                'vae_src': vae_src,
                'vae_tgt': vae_tgt,
            }
            all_samples.append(sample)

    print(f"Total valid samples: {len(all_samples)} (skipped {skipped_empty} with empty prompt)")

    # 分别检查 T5 和 VAE 是否已存在，构建待处理列表
    pending_t5 = []
    pending_vae = []
    t5_exists_count = 0
    vae_exists_count = 0

    for sample in all_samples:
        t5_ok = os.path.exists(sample['t5_seq']) and os.path.exists(sample['t5_mask'])
        vae_ok = os.path.exists(sample['vae_src']) and os.path.exists(sample['vae_tgt'])

        if not t5_ok:
            pending_t5.append(sample)
        else:
            t5_exists_count += 1

        if not vae_ok:
            pending_vae.append(sample)
        else:
            vae_exists_count += 1

    print(f"T5 features already exist for {t5_exists_count} samples, will extract for {len(pending_t5)} samples.")
    print(f"VAE latents already exist for {vae_exists_count} samples, will extract for {len(pending_vae)} samples.")

    # ---------- 提取 T5 特征 ----------
    if pending_t5:
        print("Extracting T5 features...")
        # 准备待提取的提示词列表和对应样本（用于保存）
        t5_prompts = [s['prompt'] for s in pending_t5]
        # 批量提取
        for i in tqdm(range(0, len(t5_prompts), batch_size)):
            batch_prompts = t5_prompts[i:i + batch_size]
            batch_samples = pending_t5[i:i + batch_size]
            hidden_states, masks = extract_t5_features(batch_prompts, tokenizer, text_encoder, device)

            for j, sample in enumerate(batch_samples):
                os.makedirs(os.path.dirname(sample['t5_seq']),exist_ok=True)
                os.makedirs(os.path.dirname(sample['t5_mask']), exist_ok=True)
                np.save(sample['t5_seq'], hidden_states[j])
                np.save(sample['t5_mask'], masks[j])
        print("T5 extraction finished.")
    else:
        print("All T5 features already exist, skipping T5 extraction.")

    # ---------- 提取 VAE 潜变量 ----------
    if pending_vae:
        print("Extracting VAE latents...")
        skipped_vae = 0
        for sample in tqdm(pending_vae):
            # 提取源
            src_latent, src_flip = extract_vae_latent(sample['src'], vae, transform, device)
            if src_latent is None:
                skipped_vae += 1
                continue
            # 提取目标
            tgt_latent, tgt_flip = extract_vae_latent(sample['tgt'], vae, transform, device)
            if tgt_latent is None:
                skipped_vae += 1
                continue

            # 确保目录存在
            os.makedirs(os.path.dirname(sample['vae_src']), exist_ok=True)
            os.makedirs(os.path.dirname(sample['vae_tgt']), exist_ok=True)

            np.savez_compressed(sample['vae_src'], latent=src_latent, latent_flip=src_flip)
            np.savez_compressed(sample['vae_tgt'], latent=tgt_latent, latent_flip=tgt_flip)
        print(f"VAE extraction finished. Skipped {skipped_vae} samples due to errors.")
    else:
        print("All VAE latents already exist, skipping VAE extraction.")

    print("All done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract VAE latents and T5 features with resume support.')
    parser.add_argument('--image_root', type=str, default="E:/dataset/pico-banana-400k",
                        help='Root directory containing all images.')
    parser.add_argument('--jsonl', type=str, default="E:/dataset/pico-banana-400k/sft_with_local_source_image_path.jsonl",
                        help='JSONL file path (each line a JSON object).')
    parser.add_argument('--output_dir', type=str,
                        default="E:/dataset/pico-banana-400k/preprocessed_all_moe",
                        help='Output directory for extracted features.')
    parser.add_argument('--t5_path', type=str, default='E:/AImodel/t5-base',
                        help='Path to T5 model.')
    parser.add_argument('--vae_path', type=str, default="E:/AImodel/ProMoE/sd-vae-ft-mse",
                        help='Path to VAE model.')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for center cropping.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for T5 feature extraction.')
    args = parser.parse_args()

    main(args.image_root, args.jsonl, args.output_dir, args.t5_path, args.vae_path,
         args.image_size, args.batch_size)