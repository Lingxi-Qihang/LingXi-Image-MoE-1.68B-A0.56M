import os
import os.path as osp
import torch
import torch.cuda.amp as amp
import numpy as np
import logging
from PIL import Image
from torchvision import transforms
from config import cfg
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
import argparse
import yaml
import colorlog
from diffusers.models import AutoencoderKL
from train_GradientAccumulationSteps import model_dict
import glob
from tqdm import tqdm
from utils import deep_update, find_free_port
from transformers import T5EncoderModel, T5Tokenizer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# ------------------------- 辅助函数 -------------------------
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


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s[%(asctime)s-%(levelname)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan', 'INFO': 'blue', 'WARNING': 'yellow',
            'ERROR': 'red', 'CRITICAL': 'bold_red',
        }
    )
    file_handler = logging.FileHandler(os.path.join(output_dir, "sample.log"))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def load_specific_checkpoints(checkpoint_dir, specific_steps):
    specific_steps = set(specific_steps)
    all_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'ckpt_step_*.pth'))
    specific_checkpoints = []
    step_to_checkpoint = {}
    for checkpoint in all_checkpoints:
        step_str = os.path.basename(checkpoint).split('_')[-1].replace('.pth', '')
        try:
            step = int(step_str)
            if step in specific_steps:
                specific_checkpoints.append(checkpoint)
                step_to_checkpoint[checkpoint] = step
        except ValueError:
            continue
    if not specific_checkpoints:
        logging.info(f'No checkpoints found for specified steps: {specific_steps}')
        return [], []
    sorted_pairs = sorted(specific_checkpoints, key=lambda x: step_to_checkpoint[x])
    sorted_checkpoints = [checkpoint for checkpoint in sorted_pairs]
    sorted_steps = [step_to_checkpoint[checkpoint] for checkpoint in sorted_pairs]
    logging.info(f'Found {len(sorted_checkpoints)} checkpoints for specified steps: {specific_steps}')
    return sorted_checkpoints, sorted_steps


def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))
    return sigma


def retrieve_timesteps(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def encode_image_to_latent(image_path, vae, transform, device):
    # """将图像编码为缩放后的潜变量 (1, 4, H, W)"""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.parameters

    return latent  # (1, 4, H, W)


def main(**kwargs):
    deep_update(cfg, kwargs)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = find_free_port()

    cfg.output_dir = osp.join(cfg.output_dir, cfg.model_name, cfg.custom_cfg_name)
    cfg.pmi_rank = int(os.getenv('RANK', 0))
    cfg.pmi_world_size = 1
    cfg.gpus_per_machine = torch.cuda.device_count()
    cfg.world_size = 1
    worker(0, cfg)


@torch.no_grad
def worker(gpu, cfg):
    cfg.gpu = gpu
    torch.cuda.set_device(gpu)

    setup_logging(cfg.output_dir)

    # 1. 加载 VAE
    logging.info('Initializing VAE...')
    vae = AutoencoderKL.from_pretrained(cfg.sd_vae_ft_mse_vae_path).eval().to(gpu)
    latent_shape = (cfg.image_size // 8, cfg.image_size // 8)

    # 图像预处理 (与训练完全一致)
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. 加载文本编码器 T5
    logging.info('Initializing Text Encoder (T5)...')
    t5_path = cfg.get('t5_model_path', '/data/coding/t5-base')
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    text_encoder = T5EncoderModel.from_pretrained(t5_path).eval().to(gpu)

    sample_prompts = [cfg.sample_prompts]
    logging.info(f"Using {len(sample_prompts)} prompts for sampling.")
    num_prompts = len(sample_prompts)

    # 4. 源图像处理 (在线编码，自动识别任务)
    source_image_path = cfg.get('source_image_path', None)  # 新参数：源图像路径
    if source_image_path and osp.exists(source_image_path):
        task_type = 'edit'
        source_latent = encode_image_to_latent(source_image_path, vae, transform, gpu)
        # 源图像也需要乘以 scaling_factor（预处理时已乘过，这里确认一致性）
        source_latent = source_latent * 0.18215
        logging.info(f"Edit mode: loaded source latent shape {source_latent.shape}")
    else:
        task_type = 't2i'
        source_latent = torch.zeros(1, 4, *latent_shape, device=gpu)
        logging.info(f"T2I mode: using black latent as source.")

    # 无条件分支的源图像 (全零黑图)
    null_source = torch.zeros(1, 4, *latent_shape, device=gpu)

    # 5. 加载模型checkpoint
    checkpoint_dir = osp.join(cfg.output_dir, 'checkpoints')
    val_loss_model, val_loss_model_steps = load_specific_checkpoints(checkpoint_dir, cfg.step_list_for_sample)
    guide_scales = cfg.guide_scale_list if hasattr(cfg, 'guide_scale_list') and cfg.guide_scale_list else [cfg.guide_scale]

    for ckpt_path, ckpt_step in zip(val_loss_model, val_loss_model_steps):
        for current_guide_scale in guide_scales:
            folder_name = f"img{cfg.image_size}_cfg{current_guide_scale}_seed{cfg.global_seed}_{task_type}"
            cfg.sample_folder_dir = osp.join(cfg.output_dir, 'sample', f'step{ckpt_step}', folder_name)
            os.makedirs(cfg.sample_folder_dir, exist_ok=True)
            logging.info(f"Saving samples to {cfg.sample_folder_dir} with guide_scale={current_guide_scale}")

            # 加载模型
            model_class, config_name = model_dict[cfg.model_name]
            model_cfg = getattr(cfg, config_name)
            model = model_class(**model_cfg).to(gpu)
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint['ema_model_state_dict'], strict=False)
            model.eval()

            # 获取 null_text_embed
            if hasattr(model, 'null_text_embed'):
                null_global = model.null_text_embed
            else:
                null_global = torch.zeros(1, 768, device=gpu)
                logging.warning("Model has no null_text_embed. Using zeros.")

            null_seq = null_global.unsqueeze(1).expand(-1, 512, -1)
            null_mask = torch.zeros(1, 512, dtype=torch.bool, device=gpu)

            # 开始采样
            total_samples = min(len(sample_prompts), cfg.num_fid_samples)
            for idx in tqdm(range(total_samples), desc=f"Sampling step {ckpt_step}"):
                prompt = sample_prompts[idx % num_prompts]

                # 编码文本
                inputs = tokenizer(prompt, return_tensors="pt", padding="max_length",
                                   max_length=512, truncation=True)
                input_ids = inputs.input_ids.to(gpu)
                text_mask = inputs.attention_mask.bool().to(gpu)
                t5_outputs = text_encoder(input_ids=input_ids, attention_mask=text_mask)
                text_seq = t5_outputs.last_hidden_state  # [1, 512, 768]

                arg_c = {
                    'text_sequence': text_seq,
                    'text_mask': text_mask,
                    'source_latent': source_latent,
                    'use_gradient_checkpointing': False,
                }
                arg_null = {
                    'text_sequence': null_seq,
                    'text_mask': null_mask,
                    'source_latent': null_source,
                    'use_gradient_checkpointing': False,
                }

                noise = torch.randn(1, 4, 1, *latent_shape, device=gpu)
                latents = noise

                scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=cfg.shift)
                sampling_sigmas = get_sampling_sigmas(cfg.sample_steps, cfg.sample_shift)
                timesteps, _ = retrieve_timesteps(scheduler, device=gpu, sigmas=sampling_sigmas)

                for t in timesteps:
                    latent_model_input = latents
                    timestep = torch.tensor([t] * len(latents), device=gpu)

                    noise_pred_cond = model(latent_model_input, timestep, **arg_c)
                    if isinstance(noise_pred_cond, tuple):
                        noise_pred_cond = noise_pred_cond[0]

                    if current_guide_scale > 1.0:
                        noise_pred_uncond = model(latent_model_input, timestep, **arg_null)
                        if isinstance(noise_pred_uncond, tuple):
                            noise_pred_uncond = noise_pred_uncond[0]
                        noise_pred = noise_pred_uncond + current_guide_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_cond

                    if noise_pred.shape[1] != latents.shape[1]:
                        noise_pred, _ = noise_pred.chunk(2, dim=1)

                    latents = scheduler.step(noise_pred.unsqueeze(2), t, latents, return_dict=False)[0]

                # 解码
                samples = vae.decode(latents.squeeze(2) / 0.18215).sample
                samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
                sample = samples[0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)

                img = Image.fromarray(sample)
                safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt)[:50]
                img.save(osp.join(cfg.sample_folder_dir, f"{idx:04d}_{safe_prompt}.png"))

    logging.info("Sampling completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample for T2I/Edit MoE')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--step_list_for_sample', type=str, default="5000", help='Checkpoint steps to sample')
    parser.add_argument('--guide_scale_list', type=str, default="4.0", help='CFG scales')
    parser.add_argument('--num_fid_samples', type=int, default=4)
    parser.add_argument('--prompt_file', type=str, default=None, help='File containing prompts, one per line')
    parser.add_argument('--source_image_path', type=str, default=None, help='Path to source image (for edit task)')
    parser.add_argument('--sample_prompts', type=str, default=None, help='prompts')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        custom_cfg = yaml.safe_load(file)

    custom_cfg['custom_cfg_name'] = osp.splitext(osp.basename(args.config))[0]
    custom_cfg['step_list_for_sample'] = [int(x) for x in args.step_list_for_sample.split(',')]
    custom_cfg['guide_scale_list'] = [float(x) for x in args.guide_scale_list.split(',')]
    custom_cfg['num_fid_samples'] = args.num_fid_samples
    custom_cfg['source_image_path'] = args.source_image_path
    custom_cfg['sample_prompts'] = args.sample_prompts
    if args.prompt_file:
        custom_cfg['prompt_file'] = args.prompt_file

    main(**custom_cfg)
