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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def center_crop_arr(pil_image, image_size):
    """中心裁剪"""
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
    """
    提取单张图像的 VAE 潜变量，同时返回翻转后的潜变量。
    如果图像无法打开或处理失败，返回 (None, None)。
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        latent = vae.encode(img_tensor).latent_dist.parameters
        latent_flip = vae.encode(img_tensor.flip(dims=[3])).latent_dist.parameters

        return latent.squeeze(0).detach().cpu().numpy(), latent_flip.squeeze(0).detach().cpu().numpy()
    except:
        return None, None


def extract_t5_features(prompts, tokenizer, text_encoder, device, max_length=512):
    """批量提取 T5 文本特征"""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    hidden_states = outputs.last_hidden_state.cpu().float().numpy()
    masks = inputs.attention_mask.cpu().bool().numpy()
    return hidden_states, masks


def main(image_root, json_path, output_dir, t5_path, vae_path, image_size=256, batch_size=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 加载 T5
    print("Loading T5...")
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    text_encoder = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.bfloat16).eval().to(device)

    # 加载 VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(vae_path).eval().to(device)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples in JSON: {len(data)}")

    # 输出目录
    text_output_dir = os.path.join(output_dir, 't5_text_features')
    latent_output_dir = os.path.join(output_dir, 'latents')
    os.makedirs(text_output_dir, exist_ok=True)
    os.makedirs(latent_output_dir, exist_ok=True)

    # 构建任务信息列表，并过滤掉可能存在的无效条目（如图像路径为空等）
    all_prompts = []
    all_image_paths = []
    skipped_empty = 0

    for item in data:
        prompt = item.get('input_prompt', '')
        if not prompt:
            continue  # 没有提示词，跳过
        if 'input_image' in item:
            if len(item['input_image']) == 1:
                img_name = item['input_image'][0]
                src_img_path = os.path.join(image_root, img_name)
                tgt_img_name = item['output_image']
                tgt_img_path = os.path.join(image_root, tgt_img_name)
                all_image_paths.append({
                    'task': 'edit',
                    'src': src_img_path,
                    'tgt': tgt_img_path,
                    'prompt': prompt
                })
                all_prompts.append(prompt)
        else:
            img_name = item['output_image']
            tgt_img_path = os.path.join(image_root, img_name)
            all_image_paths.append({
                'task': 't2i',
                'tgt': tgt_img_path,
                'prompt': prompt
            })
            all_prompts.append(prompt)

    print(f"Valid samples after filtering: {len(all_prompts)}")

    # 批量提取 T5 特征
    print("Extracting T5 features...")
    for i in tqdm(range(0, len(all_prompts), batch_size)):
        batch_prompts = all_prompts[i:i + batch_size]
        batch_info = all_image_paths[i:i + batch_size]

        hidden_states, masks = extract_t5_features(batch_prompts, tokenizer, text_encoder, device)

        for j, info in enumerate(batch_info):
            # 以目标图像文件名为基准
            tgt_base = os.path.splitext(os.path.basename(info['tgt']))[0]
            seq_path = os.path.join(text_output_dir, tgt_base + '_seq.npy')
            mask_path = os.path.join(text_output_dir, tgt_base + '_mask.npy')
            np.save(seq_path, hidden_states[j])
            np.save(mask_path, masks[j])

    # 提取 VAE 潜变量
    print("Extracting VAE latents...")
    skipped_vae = 0

    for info in tqdm(all_image_paths):
        if info['task'] == 'edit':
            # 编辑任务：提取源图像和目标图像的潜变量
            src_latent, src_latent_flip = extract_vae_latent(info['src'], vae, transform, device)
            if src_latent is None:
                skipped_vae += 1
                continue  # 跳过这个样本，不保存任何潜变量

            tgt_latent, tgt_latent_flip = extract_vae_latent(info['tgt'], vae, transform, device)
            if tgt_latent is None:
                skipped_vae += 1
                continue  # 跳过这个样本

            src_base = os.path.splitext(os.path.basename(info['src']))[0]
            src_save_dir = os.path.join(latent_output_dir, 'source')
            os.makedirs(src_save_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(src_save_dir, f"{src_base}.latent.npz"),
                latent=src_latent,
                latent_flip=src_latent_flip
            )

            tgt_base = os.path.splitext(os.path.basename(info['tgt']))[0]
            tgt_save_dir = os.path.join(latent_output_dir, 'target')
            os.makedirs(tgt_save_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(tgt_save_dir, f"{tgt_base}.latent.npz"),
                latent=tgt_latent,
                latent_flip=tgt_latent_flip
            )
        else:
            # 文生图任务：只提取目标图像的潜变量
            tgt_latent, tgt_latent_flip = extract_vae_latent(info['tgt'], vae, transform, device)
            if tgt_latent is None:
                skipped_vae += 1
                continue

            tgt_base = os.path.splitext(os.path.basename(info['tgt']))[0]
            tgt_save_dir = os.path.join(latent_output_dir, 'target')
            os.makedirs(tgt_save_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(tgt_save_dir, f"{tgt_base}.latent.npz"),
                latent=tgt_latent,
                latent_flip=tgt_latent_flip
            )

    print(f"Done! Skipped {skipped_vae} images during VAE latent extraction.")
    print(f"Text features saved to {text_output_dir}")
    print(f"Latents saved to {latent_output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract VAE latents and T5 features for I2I/T2I tasks.')
    parser.add_argument('--image_root', type=str, default="E:/dataset/DiffSynth-Studio/OpenGPT-4o-Image",
                        help='Root directory containing all images.')
    parser.add_argument('--json', type=str, default="E:/dataset/DiffSynth-Studio/OpenGPT-4o-Image/editing.json",
                        help='JSON file path.')
    parser.add_argument('--output_dir', type=str,
                        default="E:/dataset/DiffSynth-Studio/OpenGPT-4o-Image/preprocessed_all_moe",
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

    main(args.image_root, args.json, args.output_dir, args.t5_path, args.vae_path, args.image_size, args.batch_size)