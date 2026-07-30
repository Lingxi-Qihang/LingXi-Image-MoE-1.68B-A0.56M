import os
import os.path as osp
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.cuda.amp as amp
import torch.optim as optim
import numpy as np
import logging
import datetime
import copy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import math
from PIL import Image
from torchvision import transforms
from einops import rearrange
from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import colorlog
import glob
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from utils import deep_update, find_free_port
from torch.nn.utils import clip_grad_norm_
import warnings
warnings.filterwarnings("ignore")

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["DISTUTILS_USE_SDK"] = "1"

from config import cfg
from models.models_DiT import DiT as DiT
from models.models_TCDiT import DiT as TCDiT
from models.models_ECDiT import DiT as ECDiT
from models.models_DiffMoE import DiT as DiffMoE
from models.models_ProMoE_TC import DiT as ProMoE_TC
from models.models_ProMoE_EC import DiT as ProMoE_EC

model_dict = {
    "DiT_B": (DiT, "DiT_B_config"),
    "DiT_L": (DiT, "DiT_L_config"),
    "DiT_XL": (DiT, "DiT_XL_config"),
    "TCDiT_L_E8": (TCDiT, "TCDiT_L_E8_config"),
    "ECDiT_L_E8": (ECDiT, "ECDiT_L_E8_config"),
    "DiffMoE_B_E8": (DiffMoE, "DiffMoE_DiT_B_E8_config"),
    "DiffMoE_L_E8": (DiffMoE, "DiffMoE_DiT_L_E8_config"),
    "DiffMoE_XL_E8": (DiffMoE, "DiffMoE_DiT_XL_E8_config"),
    "ProMoE_TC_S": (ProMoE_TC, "DiT_S_config"),
    "ProMoE_TC_B": (ProMoE_TC, "DiT_B_config"),
    "ProMoE_TC_L": (ProMoE_TC, "DiT_L_config"),
    "ProMoE_TC_XL": (ProMoE_TC, "DiT_XL_config"),
    "ProMoE_EC_L": (ProMoE_EC, "DiT_L_config"),
}

# ================================================================
# 统一的序列共生 Dataset
# ================================================================
class UnifiedSymbiosisDataset(Dataset):
    def __init__(self, latent_root, text_root, image_size=256):
        self.latent_root = latent_root
        self.text_root = text_root
        self.latent_shape = (4, 1, image_size // 8, image_size // 8)
        self.bad_indices = set()  # 记录损坏文件的索引

        # 收集所有目标潜变量文件
        self.target_paths = []
        target_dir = osp.join(latent_root, "target")
        for root, dirs, files in os.walk(target_dir):
            for f in files:
                if f.endswith('.latent.npz'):
                    self.target_paths.append(osp.join(root, f))

        logging.info(f"Found {len(self.target_paths)} target latent files")

        # 初始化占位形状：尝试从第一个正常样本中获取
        self._init_shapes()
        # 预加载一个干净样本作为最终 fallback
        self._init_fallback_sample()

    def _init_shapes(self):
        """从第一个正常样本中获取潜变量和文本特征的形状"""
        for idx in range(min(100, len(self.target_paths))):
            try:
                target_path = self.target_paths[idx]
                npz_data = np.load(target_path)
                target_latent = torch.from_numpy(npz_data['latent'])
                self.latent_shape = target_latent.shape  # 如 [4, 32, 32]

                base_name = osp.splitext(osp.basename(target_path))[0].split(".latent")[0]
                seq_path = osp.join(self.text_root, base_name + "_seq.npy")
                if osp.exists(seq_path):
                    text_seq = torch.from_numpy(np.load(seq_path)).float()
                    self.text_shape = text_seq.shape  # 如 [L, 768]
                else:
                    self.text_shape = (512, 768)
                return
            except Exception:
                continue

        # 【修复】如果前100个都加载失败，抛出 RuntimeError，程序直接终止，提示用户检查数据
        raise RuntimeError("Could not determine shapes from the first 100 samples. Please check your data files!")

    def _init_fallback_sample(self):
        """预加载一个干净样本，作为所有重试失败后的最终兜底"""
        for idx in range(len(self.target_paths)):
            try:
                target_path = self.target_paths[idx]
                npz_data = np.load(target_path)
                if torch.rand(1) < 0.5:
                    target_latent = torch.from_numpy(npz_data['latent'])
                else:
                    target_latent = torch.from_numpy(npz_data['latent_flip'])

                if torch.isnan(target_latent).any() or torch.isinf(target_latent).any():
                    continue
                if target_latent.abs().max() > 20.0:
                    continue

                rel = osp.relpath(target_path, osp.join(self.latent_root, "target"))
                source_path = osp.join(self.latent_root, "source", rel)
                if osp.exists(source_path):
                    npz_src = np.load(source_path)
                    source_latent = torch.from_numpy(npz_src['latent'])
                else:
                    source_latent = torch.zeros_like(target_latent)

                base_name = osp.splitext(osp.basename(target_path))[0].split(".latent")[0]
                seq_path = osp.join(self.text_root, base_name + "_seq.npy")
                mask_path = osp.join(self.text_root, base_name + "_mask.npy")

                if osp.exists(seq_path):
                    text_seq = torch.from_numpy(np.load(seq_path)).float()
                    text_mask = torch.from_numpy(np.load(mask_path)).bool()
                    L = text_seq.shape[0]
                    if L < 512:
                        pad_seq = torch.zeros(512 - L, text_seq.shape[1])
                        pad_mask = torch.zeros(512 - L, dtype=torch.bool)
                        text_seq = torch.cat([text_seq, pad_seq], dim=0)
                        text_mask = torch.cat([text_mask, pad_mask], dim=0)
                    elif L > 512:
                        text_seq = text_seq[:512]
                        text_mask = text_mask[:512]
                else:
                    text_seq = torch.zeros(512, 768)
                    text_mask = torch.zeros(512, dtype=torch.bool)

                self.fallback_sample = (source_latent, target_latent, text_seq, text_mask)
                return
            except Exception:
                continue

        # 【修复】如果整个数据集都加载失败，抛出 RuntimeError，程序直接终止
        raise RuntimeError("Could not find any valid sample for fallback. Your entire dataset might be corrupted!")

    def __len__(self):
        return len(self.target_paths)

    def __getitem__(self, idx):
        max_retries = 5

        for attempt in range(max_retries):
            # ✅ 跳过已知的坏索引
            # 跳过已知坏索引，但最多尝试100次
            skip_attempts = 0
            while idx in self.bad_indices and skip_attempts < 100:
                idx = np.random.randint(0, len(self.target_paths))
                skip_attempts += 1

            try:
                target_path = self.target_paths[idx]
                npz_data = np.load(target_path)

                if torch.rand(1) < 0.5:
                    target_latent = torch.from_numpy(npz_data['latent'])
                else:
                    target_latent = torch.from_numpy(npz_data['latent_flip'])

                if torch.isnan(target_latent).any() or torch.isinf(target_latent).any():
                    self.bad_indices.add(idx)
                    raise ValueError("Found NaN/Inf in target latent")

                # --- 反推源潜变量路径 ---
                rel = osp.relpath(target_path, osp.join(self.latent_root, "target"))
                source_path = osp.join(self.latent_root, "source", rel)

                if osp.exists(source_path):
                    npz_src = np.load(source_path)
                    source_latent = torch.from_numpy(npz_src['latent'])
                    if torch.isnan(source_latent).any() or torch.isinf(source_latent).any():
                        raise ValueError("Found NaN/Inf in source latent")
                else:
                    source_latent = torch.zeros_like(target_latent)

                # --- 反推文本特征路径 ---
                base_name = osp.splitext(osp.basename(target_path))[0].split(".latent")[0]
                seq_path = osp.join(self.text_root, base_name + "_seq.npy")
                mask_path = osp.join(self.text_root, base_name + "_mask.npy")

                max_len = 512
                if osp.exists(seq_path):
                    text_seq = torch.from_numpy(np.load(seq_path)).float()
                    text_mask = torch.from_numpy(np.load(mask_path)).bool()
                    L = text_seq.shape[0]
                    if L < max_len:
                        pad_seq = torch.zeros(max_len - L, text_seq.shape[1])
                        pad_mask = torch.zeros(max_len - L, dtype=torch.bool)
                        text_seq = torch.cat([text_seq, pad_seq], dim=0)
                        text_mask = torch.cat([text_mask, pad_mask], dim=0)
                    elif L > max_len:
                        text_seq = text_seq[:max_len]
                        text_mask = text_mask[:max_len]
                else:
                    text_seq = torch.zeros(max_len, 768)
                    text_mask = torch.zeros(max_len, dtype=torch.bool)

                return source_latent, target_latent, text_seq, text_mask

            except Exception as e:
                logging.warning(f"Bad data at idx {idx}: {e}. Retry {attempt+1}/{max_retries}.")
                self.bad_indices.add(idx)
                idx = np.random.randint(0, len(self.target_paths))

        logging.error(f"Failed to load valid data after {max_retries} retries. Using fallback sample.")
        return self.fallback_sample


# ================================================================
# 工具函数
# ================================================================
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())
    for name, buffer in model_buffers.items():
        ema_buffers[name].copy_(buffer)


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
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"), mode='a')
    plain_formatter = logging.Formatter('[%(asctime)s-%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(plain_formatter)
    logger.addHandler(file_handler)


def save_checkpoint_single(model, ema_model, optimizer, step, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_step_{step}.pth')
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'ema_model_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    logging.info(f'Checkpoint saved at {checkpoint_path}')


def get_sigmas_timesteps(u, shift, num_train_timesteps, n_dim=4, dtype=torch.float32):
    sigma = (shift * u / (1 + (shift - 1) * u)).to(dtype=dtype)
    timesteps = (sigma * num_train_timesteps).to(dtype=dtype)
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return timesteps, sigma


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = 0.0, logit_std: float = 1.0,
    sigmoid_scale: float = 1.0, mode_scale: float = 1.29, generator=None, device='cpu'
):
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), generator=generator, device=device)
        u = u * sigmoid_scale
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), generator=generator, device=device)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), generator=generator, device=device)
    return u


# ================================================================
# 主函数
# ================================================================
def main(**kwargs):
    deep_update(cfg, kwargs)

    if 'gpu_ids' in kwargs and kwargs['gpu_ids'] is not None:
        gpu_ids = ','.join(map(str, kwargs['gpu_ids']))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"Set CUDA_VISIBLE_DEVICES to {gpu_ids}")

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = find_free_port()

    cfg.pmi_rank = int(os.getenv('RANK', 0))
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    cfg.gpus_per_machine = torch.cuda.device_count()
    cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine

    if cfg.world_size == 1:
        worker_single(cfg)
    else:
        mp.spawn(worker_multi, nprocs=cfg.gpus_per_machine, args=(cfg,))


def load_latest_checkpoint(model, ema_model, optimizer,checkpoint_dir='checkpoints', resume_checkpoint_step=None):
    if resume_checkpoint_step is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f'ckpt_step_{resume_checkpoint_step}.pth')
        if not os.path.exists(checkpoint_path):
            logging.error(f"Specified checkpoint not found: {checkpoint_path}")
            return 0
        checkpoints_to_try = [checkpoint_path]
    else:
        checkpoints_to_try = sorted(
            glob.glob(os.path.join(checkpoint_dir, 'ckpt_step_*.pth')),
            key=os.path.getmtime,
            reverse=True
        )
        if not checkpoints_to_try:
            logging.error(f"No checkpoints found in directory: {checkpoint_dir}")
            return 0

    for i, checkpoint_path in enumerate(checkpoints_to_try):
        try:
            logging.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint['model_state_dict'],
                strict=False
            )
            if len(missing_keys) > 0:
                logging.warning(f"Missing keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                logging.warning(f"Unexpected keys: {unexpected_keys}")

            if 'ema_model_state_dict' in checkpoint:
                ema_missing, _ = ema_model.load_state_dict(checkpoint['ema_model_state_dict'], strict=False)
                if len(ema_missing) > 0:
                    logging.warning(f"EMA missing keys: {ema_missing}")
                logging.info("EMA model loaded")

            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logging.info("Optimizer loaded")
                except Exception as e:
                    logging.warning(f"Failed to load optimizer state, using fresh optimizer: {e}")

            step = checkpoint.get('step', 0)
            logging.info(f'Successfully loaded checkpoint from step {step}')
            return step

        except Exception as e:
            error_msg = f"Failed to load checkpoint {checkpoint_path}: {str(e)}"
            if len(checkpoints_to_try) > 1:
                error_msg += f" (attempt {i + 1}/{len(checkpoints_to_try)})"
            logging.error(error_msg)
            if resume_checkpoint_step is not None:
                return 0

    logging.error("Could not load any checkpoint. Training from scratch.")
    return 0


def worker_single(cfg):
    gpu = 0
    torch.cuda.set_device(gpu)

    cfg.output_dir = osp.join(cfg.output_dir, cfg.model_name, cfg.custom_cfg_name)
    os.makedirs(cfg.output_dir, exist_ok=True)
    setup_logging(cfg.output_dir)

    use_amp = (cfg.param_dtype == torch.bfloat16)
    writer = SummaryWriter(log_dir=osp.join(cfg.output_dir, "tensorboard"))

    # ================================================================
    # 数据集
    # ================================================================
    img_dataset = UnifiedSymbiosisDataset(
        latent_root=cfg.latent_root,
        text_root=cfg.text_root,
        image_size=cfg.image_size,
    )

    train_batch_size = getattr(cfg, 'total_train_batch_size', 32)
    image_dataloader = DataLoader(
        img_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=cfg.img_num_workers, pin_memory=True,
        prefetch_factor=cfg.prefetch_factor, persistent_workers=True
    )
    image_rank_iter = iter(image_dataloader)

    total_images = len(img_dataset)
    steps_per_epoch = total_images // train_batch_size
    if total_images % train_batch_size != 0:
        steps_per_epoch += 1
    logging.info(f"Image Num {total_images}, steps per epoch: {steps_per_epoch}")

    # ================================================================
    # 模型初始化
    # ================================================================
    model_class, config_name = model_dict[cfg.model_name]
    model_cfg = getattr(cfg, config_name)
    model = model_class(**model_cfg).to(gpu)
    model_ema = copy.deepcopy(model).eval().requires_grad_(False)

    model_size = sum([p.numel() for p in model.parameters()]) / (1000 ** 3)
    logging.info(f'Model size: {model_size:.3f}B')

    optimizer = optim.AdamW(
        params=list(model.parameters()),
        lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay, fused=True
    )

    scaler = amp.GradScaler(enabled=False)

    cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoints')
    step = 0

    # 断点续训
    if cfg.resume_checkpoint:
        resume_step = getattr(cfg, 'resume_checkpoint_step', None)
        loaded_step = load_latest_checkpoint(
            model, model_ema, optimizer,
            cfg.checkpoint_dir, resume_step
        )
        if loaded_step > 0:
            step = loaded_step
            logging.info(f'Resumed from step {step}')
        else:
            logging.info('No checkpoint found, training from scratch.')

    model.train()
    model_ema.eval()

    # 假设你从280万步续训，目标是320万步
    remaining_steps = cfg.num_steps - step

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=remaining_steps,  # 在剩余40万步内完成衰减
        eta_min=1e-6
    )

    # ================================================================
    # 【新增】梯度累积配置
    # ================================================================
    grad_accum_steps = getattr(cfg, 'grad_accum_steps', 1)
    logging.info(f'Gradient Accumulation Steps: {grad_accum_steps}')
    logging.info(f'Effective Batch Size: {train_batch_size * grad_accum_steps}')

    # ================================================================
    # 训练循环
    # ================================================================
    logging.info('Start training loop (single GPU, Symbiosis)')
    epoch = 0
    accum_counter = 0
    optimizer.zero_grad()  # 在循环开始前清零一次
    LOSS_SKIP_ENABLED_STEP = 2850000  # 从多少步开始启用Loss跳过
    LOSS_THRESHOLD = 1.5  # Loss超过此值则跳过


    while step < cfg.num_steps:
        try:
            source_z, target_z, rank_text_seq, rank_text_mask = next(image_rank_iter)
        except StopIteration:
            epoch += 1
            logging.info("Reload dataloader")
            image_rank_iter = iter(image_dataloader)
            source_z, target_z, rank_text_seq, rank_text_mask = next(image_rank_iter)

        source_z = source_z.to(gpu, non_blocking=True)
        target_z = target_z.to(gpu, non_blocking=True)
        rank_text_seq = rank_text_seq.to(gpu, non_blocking=True)
        rank_text_mask = rank_text_mask.to(gpu, non_blocking=True)

        # 目标潜变量采样
        #posterior = DiagonalGaussianDistribution(target_z)
        #target_z_sampled = posterior.sample().mul_(0.18215)

        target_z_sampled = target_z[:, :4, :, :].mul_(0.18215)
        target_z_sampled = rearrange(target_z_sampled, "B C H W -> B C 1 H W")

        # 源潜变量（无需加噪，直接缩放）
        source_z_sampled = source_z.mul_(0.18215)
        source_z_sampled = rearrange(source_z_sampled, "B C H W -> B C 1 H W")

        # 时间步
        rank_img_u = compute_density_for_timestep_sampling(
            weighting_scheme=cfg.weighting_scheme, batch_size=len(target_z_sampled),
            logit_mean=cfg.logit_mean, logit_std=cfg.logit_std,
            sigmoid_scale=cfg.sigmoid_scale, mode_scale=cfg.mode_scale, device=gpu
        )
        rank_img_t, rank_img_sigma = get_sigmas_timesteps(rank_img_u, cfg.shift, cfg.num_train_timesteps, n_dim=4)

        t, sigmas = rank_img_t, rank_img_sigma

        # 加噪：仅对目标部分加噪
        noise = torch.randn_like(target_z_sampled)
        noised_target = (1.0 - sigmas.squeeze()).view(target_z_sampled.shape[0], 1, 1, 1, 1) * target_z_sampled + sigmas.squeeze().view(target_z_sampled.shape[0], 1, 1, 1, 1) * noise

        # 文本条件
        arg_c = {
            'text_mask': rank_text_mask,
            'text_sequence': rank_text_seq,
            'source_latent': source_z_sampled,  # 传入源潜变量，模型内部自动拼接
            'use_gradient_checkpointing': cfg.use_gradient_checkpointing,
        }

        with amp.autocast(dtype=cfg.param_dtype, enabled=use_amp):
            model_output = model(noised_target, t, global_step=step, **arg_c)

        # Loss：模型内部已移除文本和源部分，输出仅针对目标
        if isinstance(model_output, tuple):
            model_pred = model_output[0]
        else:
            model_pred = model_output

        # 处理可能的通道加倍 (learn_sigma)
        if model_pred.shape[1] != noised_target.shape[1]:
            model_pred, _ = model_pred.chunk(2, dim=1)
        model_pred = model_pred.unsqueeze(2)

        target = noise - target_z_sampled
        mse_loss = (model_pred - target) ** 2
        mse_loss = mse_loss.mean()

        # ================================================================
        # 【新增】梯度累积：Loss 除以累积步数，保持梯度尺度一致
        # ================================================================
        # if step >= LOSS_SKIP_ENABLED_STEP and mse_loss.item() > LOSS_THRESHOLD:
        #     logging.warning(f"Abnormal loss skipped at step {step}: {mse_loss.item():.4f}")
        #     torch.cuda.empty_cache()
        #     # 核心修复：用 mse_loss * 0.0 替代 torch.tensor(0.0)
        #     # 它在数值上是 0，但保留了计算图连接，backward 时会释放显存且梯度为零
        #     scaled_loss = mse_loss * 0.0
        #     scaler.scale(scaled_loss).backward()
        #     accum_counter += 1
        #else:
        scaled_loss = mse_loss / grad_accum_steps
        scaler.scale(scaled_loss).backward()
        accum_counter += 1

        # 只有累积到指定步数时才更新参数
        if accum_counter % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            update_ema(model_ema, model)

            # 【注意】step 只在优化器更新时递增，日志、保存、warmup 都基于优化步
            step += 1
            scheduler.step()
            # 日志打印
            if step % cfg.log_interval == 0:
                logging.info(f"epoch {epoch}-step {step} loss: {mse_loss.item():.6f} (accumulated {grad_accum_steps} forwards)")
            writer.add_scalar('Loss/train', mse_loss.item(), step)

            # Warmup（基于优化步）
            warmup_steps = getattr(cfg, 'warmup_steps', 1000)
            if step < warmup_steps:
                current_lr = cfg.lr * (step + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            # 保存 checkpoint
            if step != 0 and step % cfg.save_ckpt_interval == 0:
                save_checkpoint_single(model, model_ema, optimizer, step, cfg.checkpoint_dir)

    logging.info('Training completed!')
    writer.close()


def worker_multi(gpu, cfg):
    # 多卡训练入口（待实现）
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Unified Symbiosis MoE')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps (default: 1, no accumulation)')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        custom_cfg = yaml.safe_load(file)

    custom_cfg.setdefault('custom_cfg_name', osp.splitext(osp.basename(args.config))[0])
    # 将命令行参数合并到配置中
    if args.grad_accum_steps > 1:
        custom_cfg['grad_accum_steps'] = args.grad_accum_steps
    main(**custom_cfg)