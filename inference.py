import os, torch, yaml, argparse, logging, glob
import numpy as np
from PIL import Image
from tqdm import tqdm

os.environ["DISABLE_VERSION_CHECK"] = "1"
from diffusers.models import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, T5Tokenizer

from train_GradientAccumulationSteps import model_dict
from config import cfg as base_cfg

logging.basicConfig(level=logging.INFO, format='[%(asctime)s-%(levelname)s]: %(message)s')


def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return sigma


def retrieve_timesteps(scheduler, sigmas, device=None):
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    return scheduler.timesteps


def load_prompts(prompt, prompt_dir):
    if prompt_dir:
        txt_files = sorted(glob.glob(os.path.join(prompt_dir, '*.txt')))
        if not txt_files:
            raise FileNotFoundError(f'No .txt files found in {prompt_dir}')
        prompts = []
        for f in txt_files:
            with open(f, 'r', encoding='utf-8') as fh:
                content = fh.read().strip()
            name = os.path.splitext(os.path.basename(f))[0]
            prompts.append((name, content))
        logging.info(f'Loaded {len(prompts)} prompts from {prompt_dir}')
        return prompts
    else:
        return [('prompt', prompt)]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='ProMoE T2I Inference')
    parser.add_argument('--config', type=str, required=True, help='YAML config (for model architecture)')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path (.pth)')
    parser.add_argument('--prompt', type=str, default=None, help='Single text prompt')
    parser.add_argument('--prompt_dir', type=str, default=None, help='Directory of .txt prompt files')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    parser.add_argument('--image_size', type=int, default=256, help='Output image size')
    parser.add_argument('--guide_scale', type=float, default=4.0, help='CFG guidance scale')
    parser.add_argument('--sample_steps', type=int, default=50, help='Sampling steps')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--vae_path', type=str, default='E:/AImodel/ProMoE/sd-vae-ft-mse', help='VAE model path')
    parser.add_argument('--t5_path', type=str, default='E:/AImodel/t5-base', help='T5 model path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    if not args.prompt and not args.prompt_dir:
        parser.error('Either --prompt or --prompt_dir is required')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    with open(args.config,encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    latent_size = args.image_size // 8

    # 加载 VAE
    logging.info('Loading VAE...')
    vae = AutoencoderKL.from_pretrained(args.vae_path).eval().to(device)

    # 加载 T5
    logging.info('Loading T5...')
    tokenizer = T5Tokenizer.from_pretrained(args.t5_path)
    text_encoder = T5EncoderModel.from_pretrained(args.t5_path).eval().to(device)

    # 无条件分支
    null_mask = torch.zeros(1, 512, dtype=torch.bool, device=device)

    # 加载模型
    logging.info('Loading model...')
    model_class, config_name = model_dict[cfg['model_name']]
    # 从 config.py 读取完整模型配置，再用 YAML 中的字段覆盖
    model_cfg = dict(getattr(base_cfg, config_name, {}))
    model_cfg.update(cfg.get(config_name, {}))
    model_cfg['input_size'] = latent_size
    model = model_class(**model_cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt.get('ema_model_state_dict', ckpt), strict=False)
    model.eval()

    if hasattr(model, 'null_text_embed'):
        null_global = model.null_text_embed
    else:
        null_global = torch.zeros(1, 768, device=device)
    null_seq = null_global.unsqueeze(1).expand(-1, 512, -1)

    source_latent = torch.zeros(1, 4, latent_size, latent_size, device=device)
    null_source = torch.zeros(1, 4, latent_size, latent_size, device=device)

    os.makedirs(args.output, exist_ok=True)

    prompts = load_prompts(args.prompt, args.prompt_dir)
    for idx, (name, prompt_text) in enumerate(tqdm(prompts, desc='Prompts')):

        save_name = f'{name}.png'
        save_path = os.path.join(args.output, save_name)
        if os.path.exists(save_path):
            logging.info(f'[SKIP] {save_name} already exists, skipping...')
            continue
    
        torch.manual_seed(args.seed + idx)
        torch.cuda.manual_seed(args.seed + idx)

        # 每个 prompt 重新初始化调度器，避免 state 累积
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
        sampling_sigmas = get_sampling_sigmas(args.sample_steps, 1.0)
        timesteps = retrieve_timesteps(scheduler, sigmas=sampling_sigmas, device=device)

        # 编码 prompt
        inputs = tokenizer(prompt_text, return_tensors='pt', padding='max_length',
                           max_length=512, truncation=True).to(device)
        text_mask = inputs.attention_mask.bool()
        t5_outputs = text_encoder(input_ids=inputs.input_ids, attention_mask=text_mask)
        text_seq = t5_outputs.last_hidden_state

        arg_c = {'text_sequence': text_seq, 'text_mask': text_mask, 'source_latent': source_latent}
        arg_null = {'text_sequence': null_seq, 'text_mask': null_mask, 'source_latent': null_source}

        latents = torch.randn(1, 4, 1, latent_size, latent_size, device=device)

        for t in timesteps:
            timestep = torch.full((1,), t, device=device)

            noise_pred_cond = model(latents, timestep, **arg_c)
            if isinstance(noise_pred_cond, tuple):
                noise_pred_cond = noise_pred_cond[0]

            if args.guide_scale > 1.0:
                noise_pred_uncond = model(latents, timestep, **arg_null)
                if isinstance(noise_pred_uncond, tuple):
                    noise_pred_uncond = noise_pred_uncond[0]
                noise_pred = noise_pred_uncond + args.guide_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            if noise_pred.shape[1] != latents.shape[1]:
                noise_pred, _ = noise_pred.chunk(2, dim=1)

            latents = scheduler.step(noise_pred.unsqueeze(2), t, latents, return_dict=False)[0]

        samples = vae.decode(latents.squeeze(2) / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        img = Image.fromarray(samples[0].cpu().permute(1, 2, 0).numpy().astype(np.uint8))

        #save_name = f'{name}_cfg{args.guide_scale}_seed{args.seed + idx}.png'
        #img.save(os.path.join(args.output, save_name))
        img.save(save_path)

    logging.info(f'All done. {len(prompts)} images saved to {args.output}')


if __name__ == '__main__':
    main()
    """
    python inference.py  --config configs/joint.yaml  --ckpt  ProMoE_L/ProMoE_TC_L/joint/checkpoints/ckpt_step_400000.pth --prompt_dir "D:/mywork/pythonProject/lingxi_video/ELLA-main/ELLA-main/dpg_bench/prompts"  --output outputs
    python inference.py --config configs/004_ProMoE_B.yaml --ckpt E:/AImodel/ProMoE/ProMoE_B/ProMoE_TC_B/004_ProMoE_B/checkpoints/ckpt_step_2850000.pth --prompt "A cat holds a poster" --output outputs --vae_path E:/AImodel/ProMoE/sd-vae-ft-mse --t5_path E:/AImodel/t5-base --guide_scale 4.0  --seed 42
    """