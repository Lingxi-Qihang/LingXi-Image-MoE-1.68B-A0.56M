:: t2i

python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "A bustling vintage bookstore interior bathed in warm, golden hour sunlight streaming through tall arched windows, illuminating dust motes dancing in the air, with rows of leather-bound books lining wooden shelves, featuring a prominent hand-painted wooden sign hanging above the counter that reads "LingXi-Image-MoE" in elegant calligraphy, shot in a cinematic photorealistic style with shallow depth of field focusing on the sign and foreground books."

python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "a spectacular display of fireworks illuminates the night sky with bursts of red, white, and blue. the vibrant colors reflect off a nearby lake, creating a mirror image of the aerial spectacle. in the foreground, silhouettes of a small crowd can be seen gathered to watch the show, with some individuals pointing upwards in awe."

python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "A cat holds a poster with rainbow text \"STOP\""
python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "An anthropomorphic rainbow fox, its fur dotted with twinkling stardust, dragging a fluffy seven-color gradient tail, standing on a fantasy grassland full of glowing flowers, with floating colorful crystals in the background, bright and dreamy overall tone, full of details."
python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "A fantasy dragon, its body is dark purple gradient, its scales shine with dark gold light, its wings are covered with dark patterns, spitting dark purple flames from its mouth, surrounded by ink-colored clouds and glowing stars, with a mysterious starry sky in the background."
python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "This is a watercolor illustration of a young girl with short black hair and fair skin, wearing a straw hat adorned with a blue flower, a white blouse, and a blue pinafore dress, sitting on a wooden swing. She holds onto the swing's ropes, surrounded by lush green foliage and colorful flowers. The background is a soft, white wash, enhancing the vibrant colors of the plants. The style is whimsical and slightly impressionistic, with delicate brushstrokes and a serene, idyllic atmosphere."

python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "A beautiful girl, delicate and the half-body shot portrait, light, ultra detailed features, romantic atmosphere, gentle and ethereal mood, The warm light shines on the hair, a half-body shot, a cold and atmospheric scene, holding snowflakes, with some of the snowflakes falling on the head, and the sunlight shining on the upper left corner."

python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "Single still. a cat knocks over a fire-alarm lever; the viewer should instantly see what follows."

python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "Single frame. a cat loosens a fire-alarm lever; ensure the immediate outcome is apparent."

python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "A waitress informs a young couple that another table has secretly paid for their meal, causing the couple to look at each other in stunned, happy disbelief."

python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "Two panel comic, left to right. Left panel: a chef ignites wet leaves. Right panel: which leads to, angry bees swarm."

python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "A grid of four images. Top left: a cat. Top right: a dog. Bottom left: a bird. Bottom right: a fish."

:: i2i
python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "Remove the word 'SALAD' at the top of the chalkboard."  --source_image_path  E:/dataset/DiffSynth-Studio/OpenGPT-4o-Image/cat_i2i/input_0.png

python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "Remove the word 'LEMON' from the image."  --source_image_path  E:/dataset/DiffSynth-Studio/OpenGPT-4o-Image/cat_i2i/input_1.png


python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "Change the bowl to a clear glass material."  --source_image_path  E:/dataset/DiffSynth-Studio/OpenGPT-4o-Image/cat_i2i/input_41324.jpg

python sample.py --config configs/joint.yaml --step_list_for_sample 1500000 --guide_scale_list 4.0 --num_fid_samples 5  --sample_prompts "Change the floor's material to polished oak wood."  --source_image_path  E:/dataset/DiffSynth-Studio/OpenGPT-4o-Image/cat_i2i/input_41302.jpg

pause