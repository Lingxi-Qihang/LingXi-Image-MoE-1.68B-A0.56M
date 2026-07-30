"""ProMoE diffusers 管道组件：ProMoEDiTModel + ProMoEPipeline"""
import os, json
import torch
import torch.nn as nn
from PIL import Image
from diffusers import DiffusionPipeline, ModelMixin, ConfigMixin
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME
from safetensors.torch import load_file, save_file

from config import cfg as base_cfg
from train_GradientAccumulationSteps import model_dict


class ProMoEDiTModel(ModelMixin, ConfigMixin):
    """将原生 ProMoE DiT 包装为 diffusers 兼容模型"""
    config_name = "promoe_dit_config.json"

    def __init__(self, model_cfg: dict):
        super().__init__()
        self.register_to_config(**model_cfg)
        model_class, _ = model_dict[model_cfg["model_name"]]
        skip_keys = {"model_name", "_class_name", "_diffusers_version", "_name_or_path"}
        model_kwargs = {k: v for k, v in model_cfg.items() if k not in skip_keys}
        self.dit = model_class(**model_kwargs)
        self._internal_dict = model_cfg

    def forward(self, x, timestep, **kwargs):
        return self.dit(x, timestep, **kwargs)

    @classmethod
    def from_config(cls, config_dict: dict, **kwargs):
        merged = {**config_dict, **kwargs}
        return cls(merged)

    def _save_to_safetensors(self, save_dir):
        state_dict = self.dit.state_dict()
        weights = {k: v.contiguous().to("cpu") for k, v in state_dict.items()}
        save_file(weights, os.path.join(save_dir, SAFETENSORS_WEIGHTS_NAME))

    @classmethod
    def _load_from_safetensors(cls, weights_path, config):
        model = cls(config)
        state_dict = load_file(weights_path)
        model.dit.load_state_dict(state_dict, strict=False)
        return model


class ProMoEPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->dit_model->vae"
    _optional_components = []

    def __init__(self, vae, text_encoder, tokenizer, dit_model, scheduler):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            dit_model=dit_model,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 8
        self.null_text_embed = dit_model.dit.null_text_embed

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] = None,
        prompt_embeds: torch.Tensor = None,
        negative_prompt_embeds: torch.Tensor = None,
        generator: torch.Generator = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        height: int = 256,
        width: int = 256,
        output_type: str = "pil",
        return_dict: bool = True,
    ):
        device = self.device

        if prompt_embeds is None:
            if isinstance(prompt, str):
                prompt = [prompt]
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding="max_length",
                max_length=512, truncation=True,
            ).to(device)
            text_mask = inputs.attention_mask.bool()
            t5_out = self.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=text_mask,
            )
            prompt_embeds = t5_out.last_hidden_state
        else:
            text_mask = torch.ones(
                prompt_embeds.shape[0], 512, dtype=torch.bool, device=device,
            )

        B = prompt_embeds.shape[0]
        latent_h, latent_w = height // 8, width // 8

        if negative_prompt_embeds is None:
            null_global = self.null_text_embed.to(device)
            negative_prompt_embeds = null_global.unsqueeze(0).expand(B, 1, -1)
            negative_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            negative_prompt_embeds = negative_prompt_embeds.expand(-1, 512, -1)
            negative_mask = negative_mask.expand(-1, 512)
        else:
            negative_mask = text_mask

        latents = torch.randn(
            B, 4, 1, latent_h, latent_w, device=device, generator=generator,
        )

        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=1.0,
        )
        sampling_sigmas = self._get_sampling_sigmas(num_inference_steps, 1.0)
        self.scheduler.set_timesteps(sigmas=sampling_sigmas, device=device)
        timesteps = self.scheduler.timesteps

        source_latent = torch.zeros(B, 4, latent_h, latent_w, device=device)
        null_source = torch.zeros(B, 4, latent_h, latent_w, device=device)

        for t in timesteps:
            timestep = torch.full((B,), t, device=device)

            arg_c = {
                "text_sequence": prompt_embeds,
                "text_mask": text_mask,
                "source_latent": source_latent,
            }
            arg_null = {
                "text_sequence": negative_prompt_embeds,
                "text_mask": negative_mask,
                "source_latent": null_source,
            }

            noise_pred_cond = self.dit_model(latents, timestep, **arg_c)
            if isinstance(noise_pred_cond, tuple):
                noise_pred_cond = noise_pred_cond[0]

            if guidance_scale > 1.0:
                noise_pred_uncond = self.dit_model(latents, timestep, **arg_null)
                if isinstance(noise_pred_uncond, tuple):
                    noise_pred_uncond = noise_pred_uncond[0]
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = noise_pred_cond

            if noise_pred.shape[1] != latents.shape[1]:
                noise_pred, _ = noise_pred.chunk(2, dim=1)

            latents = self.scheduler.step(
                noise_pred.unsqueeze(2), t, latents, return_dict=False
            )[0]

        latents = latents.squeeze(2) / 0.18215
        images = self.vae.decode(latents).sample
        images = torch.clamp(127.5 * images + 128.0, 0, 255).to(torch.uint8)

        if output_type == "pil":
            pil_images = []
            for img in images:
                pil_images.append(
                    Image.fromarray(img.cpu().permute(1, 2, 0).numpy())
                )
            return pil_images
        return images

    @staticmethod
    def _get_sampling_sigmas(sampling_steps, shift):
        import numpy as np
        sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
        sigma = shift * sigma / (1 + (shift - 1) * sigma)
        return sigma
