import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer
import argparse


def extract_t5_features(json_path, image_root, output_dir, t5_path, batch_size=8):
    tokenizer = T5Tokenizer.from_pretrained(t5_path)
    text_encoder = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.bfloat16).cuda().eval()

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompts = []
    image_paths = []
    for item in data:
        prompt = item['input_prompt']
        img_name = item['output_image']
        img_path = os.path.join(image_root, img_name)
        prompts.append(prompt)
        image_paths.append(img_path)

    print(f"Total samples: {len(prompts)}")

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_paths = image_paths[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to('cuda')

        with torch.no_grad():
            outputs = text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            # 保留完整序列 [B, L, 768]
            hidden_states = outputs.last_hidden_state.cpu().float().numpy()
            # 保存 attention_mask 用于还原真实长度
            masks = inputs.attention_mask.cpu().bool().numpy()

        for j, img_path in enumerate(batch_paths):
            relative_path = os.path.relpath(img_path, image_root)
            base_path = os.path.splitext(relative_path)[0]

            # 保存完整序列和掩码
            save_seq_path = os.path.join(output_dir, base_path + '_seq.npy')
            save_mask_path = os.path.join(output_dir, base_path + '_mask.npy')
            os.makedirs(os.path.dirname(save_seq_path), exist_ok=True)

            np.save(save_seq_path, hidden_states[j])
            np.save(save_mask_path, masks[j])

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default="E:/dataset/DiffSynth-Studio/OpenGPT-4o-Image/gen.json",
                        help='Prompt JSON file')
    parser.add_argument('--image_root', type=str, default="E:/dataset/DiffSynth-Studio/OpenGPT-4o-Image/",
                        help='Image root directory')
    parser.add_argument('--output', type=str, default="E:/dataset/DiffSynth-Studio/OpenGPT-4o-Image/t5_text_features",
                        help='Output directory for .npy files')
    parser.add_argument('--t5_path', type=str, default='E:/AImodel/t5-base')
    args = parser.parse_args()

    extract_t5_features(args.json, args.image_root, args.output, args.t5_path)