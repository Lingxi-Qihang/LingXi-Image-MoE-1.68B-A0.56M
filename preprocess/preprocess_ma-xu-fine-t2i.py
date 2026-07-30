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
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ================== 全局函数：中心裁剪 ==================
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


# ================== 数据集类（返回两个尺寸的张量） ==================
class DualSizeImageDataset(Dataset):
    def __init__(self, image_paths, transform_256, transform_512, max_retry=100):
        self.image_paths = image_paths
        self.transform_256 = transform_256
        self.transform_512 = transform_512
        self.max_retry = min(max_retry, len(image_paths))  # 最多重试100次

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        attempts = 0
        current_idx = idx

        while attempts < self.max_retry:
            img_path = self.image_paths[current_idx]
            try:
                img = Image.open(img_path).convert("RGB")

                # 256 尺寸
                cropped_256 = center_crop_arr(img, 256)
                tensor_256 = self.transform_256(cropped_256)

                # 512 尺寸
                cropped_512 = center_crop_arr(img, 512)
                tensor_512 = self.transform_512(cropped_512)

                # 成功加载，返回正常数据和原始索引
                return tensor_256, tensor_512, current_idx, img_path

            except Exception:
                # 当前图片损坏，随机换一张
                current_idx = torch.randint(0, len(self), (1,)).item()
                attempts += 1



# ================== T5 特征提取 ==================
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


# ================== 合并版 VAE 潜变量提取（一次返回两个尺寸） ==================
def process_vae_latents_combined(
    image_paths,
    vae,
    transform_256,
    transform_512,
    latent_output_dir,
    vae_batch_size,
    num_workers,
    device
):
    """
    一次遍历图片，同时生成 256 和 512 两个尺寸的潜变量并保存。
    所有图片处理完毕后统一删除原始文件。
    """
    dataset = DualSizeImageDataset(image_paths, transform_256, transform_512)
    dataloader = DataLoader(
        dataset,
        batch_size=vae_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    save_dir_256 = os.path.join(latent_output_dir, "target")
    save_dir_512 = os.path.join(latent_output_dir, "target_512")
    os.makedirs(save_dir_256, exist_ok=True)
    os.makedirs(save_dir_512, exist_ok=True)

    skipped = 0
    print(f"[VAE] Combined processing (256+512), saving to {save_dir_256} & {save_dir_512}, workers={num_workers}")

    for batch_256, batch_512, batch_indices, batch_paths in tqdm(dataloader):
        # 过滤掉无效路径
        valid_mask = [p is not None for p in batch_paths]
        if not any(valid_mask):
            skipped += len(batch_256)
            continue

        # 256 编码
        valid_256 = batch_256[valid_mask].to(device)
        valid_paths = [p for p, m in zip(batch_paths, valid_mask) if m]

        with torch.no_grad():
            latents_256 = vae.encode(valid_256).latent_dist.parameters
            latents_flip_256 = vae.encode(valid_256.flip(dims=[3])).latent_dist.parameters
        latents_256_np = latents_256.cpu().numpy()
        latents_flip_256_np = latents_flip_256.cpu().numpy()

        for latent, latent_flip, img_path in zip(latents_256_np, latents_flip_256_np, valid_paths):
            base = os.path.splitext(os.path.basename(img_path))[0]
            np.savez_compressed(
                os.path.join(save_dir_256, f"{base}.latent.npz"),
                latent=latent,
                latent_flip=latent_flip
            )

        # 512 编码
        valid_512 = batch_512[valid_mask].to(device)
        with torch.no_grad():
            latents_512 = vae.encode(valid_512).latent_dist.parameters
            latents_flip_512 = vae.encode(valid_512.flip(dims=[3])).latent_dist.parameters
        latents_512_np = latents_512.cpu().numpy()
        latents_flip_512_np = latents_flip_512.cpu().numpy()

        for latent, latent_flip, img_path in zip(latents_512_np, latents_flip_512_np, valid_paths):
            base = os.path.splitext(os.path.basename(img_path))[0]
            np.savez_compressed(
                os.path.join(save_dir_512, f"{base}.latent.npz"),
                latent=latent,
                latent_flip=latent_flip
            )

        # 删除这批 batch 中已处理的原始图片及对应 txt/json
        for img_path in valid_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
            base = os.path.splitext(img_path)[0]  # 完整路径去掉扩展名
            for ext in ['.json', '.txt']:
                aux = base + ext
                if os.path.exists(aux):
                    os.remove(aux)

    return skipped




# ================== 主函数 ==================
def main(image_root, output_dir, t5_path, vae_path, image_size=256, batch_size=64,
         vae_batch_size=16, num_workers=8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ---------- 加载 T5 ----------
    print("Loading T5...")
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    text_encoder = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.bfloat16).eval().to(device)

    # ---------- 加载 VAE ----------
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(vae_path).eval().to(device)

    # ---------- 图像预处理（只负责 ToTensor + Normalize） ----------
    transform_256 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_512 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # ---------- 扫描目录，配对图片与 txt 提示词 ----------
    print(f"Scanning image directory: {image_root}")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_prompts = []
    all_image_info = []

    for root, dirs, files in os.walk(image_root):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                img_path = os.path.join(root, file)
                txt_path = os.path.splitext(img_path)[0] + '.txt'
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()

                    if not prompt:
                        continue

                    all_image_info.append({
                        'task': 't2i',
                        'tgt': img_path,
                        'prompt': prompt
                    })
                    all_prompts.append(prompt)

    print(f"Found {len(all_prompts)} image-text pairs.")

    # ---------- 输出目录 ----------
    text_output_dir = os.path.join(output_dir, 't5_text_features')
    latent_output_dir = os.path.join(output_dir, 'latents')
    os.makedirs(text_output_dir, exist_ok=True)
    os.makedirs(latent_output_dir, exist_ok=True)

    # ==================== T5 特征提取（暂时注释） ====================
    # ==================== T5 特征提取（暂时注释） ====================
    # 批量提取 T5 特征
    # print("Extracting T5 features...")
    # for i in tqdm(range(0, len(all_prompts), batch_size)):
    #     batch_prompts = all_prompts[i:i + batch_size]
    #     batch_info = all_image_paths[i:i + batch_size]
    #
    #     hidden_states, masks = extract_t5_features(batch_prompts, tokenizer, text_encoder, device)
    #
    #     for j, info in enumerate(batch_info):
    #         tgt_base = os.path.splitext(os.path.basename(info['tgt']))[0]
    #         seq_path = os.path.join(text_output_dir, tgt_base + '_seq.npy')
    #         mask_path = os.path.join(text_output_dir, tgt_base + '_mask.npy')
    #         np.save(seq_path, hidden_states[j].astype(np.float16))
    #         np.save(mask_path, masks[j])

    # ==================== 合并版 VAE 潜变量提取 ====================
    all_tgt_paths = [info['tgt'] for info in all_image_info]

    skipped = process_vae_latents_combined(
        image_paths=all_tgt_paths,
        vae=vae,
        transform_256=transform_256,
        transform_512=transform_512,
        latent_output_dir=latent_output_dir,
        vae_batch_size=vae_batch_size,
        num_workers=num_workers,
        device=device
    )

    print(f"Done! Skipped {skipped} images during VAE latent extraction.")
    print(f"Text features saved to {text_output_dir}")
    print(f"Latents saved to {latent_output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract VAE latents and T5 features from image-txt pairs.')
    parser.add_argument('--image_root', type=str,
                        default="/data/coding/synthetic_enhanced_prompt_square_resolution_extracted_data/",
                        help='Root directory containing images and corresponding txt files.')
    parser.add_argument('--output_dir', type=str,
                        default="/data/coding/ma-xu-fine-t2i-preprocessed_all_moe/",
                        help='Output directory for extracted features.')
    parser.add_argument('--t5_path', type=str, default="/data/coding/t5-base",
                        help='Path to T5 model.')
    parser.add_argument('--vae_path', type=str, default="/data/coding/sd-vae-ft-mse",
                        help='Path to VAE model.')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for center cropping (legacy).')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for T5 feature extraction.')
    parser.add_argument('--vae_batch_size', type=int, default=64,
                        help='Batch size for VAE latent extraction.')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers for image loading.')
    args = parser.parse_args()

    main(
        image_root=args.image_root,
        output_dir=args.output_dir,
        t5_path=args.t5_path,
        vae_path=args.vae_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        vae_batch_size=args.vae_batch_size,
        num_workers=args.num_workers
    )