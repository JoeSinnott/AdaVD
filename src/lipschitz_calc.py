from template import template_dict

import lpips as lp
import torch
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as transforms
from PIL import Image
import regex as re
import glob
from tqdm import tqdm

@torch.inference_mode()
def compute_normalized_embeddings(text_list, tokenizer, text_encoder, device="cuda"):
    """
    Generate flattened, normalized CLIP embeddings for a list of strings for easy cosine similarity calculations.
    """
    inputs = tokenizer(
        text_list, 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    ).to(device)
    
    raw_embeddings = text_encoder(inputs.input_ids)[0] 
    
    flat_embeddings = raw_embeddings.view(raw_embeddings.size(0), -1)
    
    return torch.nn.functional.normalize(flat_embeddings, p=2, dim=1)

import glob
import re

def load_all_seeds_for_prompt(cat, prompt, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    clean_prompt = re.sub(r'[^\w\s]', '', prompt).replace(', ', '_')
    
    safe_prefix = clean_prompt[:30]
    
    seed_tensors = []
    for index in range(5):
        search_pattern = f"/content/drive/MyDrive/AdaVD Images and Figures/Exp1_sb_0.7/crocodile/{cat}/crocodile/original/{safe_prefix}*_{index}.png"
        
        matching_files = glob.glob(search_pattern)
        
        if not matching_files:
            raise FileNotFoundError(f"Could not find file using pattern: {search_pattern}")
            
        actual_file_path = matching_files[0]
        
        img = Image.open(actual_file_path).convert('RGB')
        seed_tensors.append(transform(img).unsqueeze(0))
        
    return torch.cat(seed_tensors, dim=0).to(device)

device = "cuda"
tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4", 
                subfolder="text_encoder", 
                torch_dtype=torch.float16
            ).to(device).eval()

loss_fc = lp.LPIPS(net = "alex").to(device)

prompt_embeddings = {}

for cat in template_dict:
    prompt_embeddings[cat] = compute_normalized_embeddings(template_dict[cat],
                                                           tokenizer,
                                                           text_encoder)

lipschitz_constants = {}

for cat, embeds in prompt_embeddings.items():
    print(f"\n--- Processing Category: {cat.upper()} ---")
    embeds = embeds.float()
    cos_sims = (embeds @ embeds.T).squeeze().cpu().numpy()
    
    images_list = []
    prompts = template_dict[cat]
    for prompt in tqdm(prompts, desc=f"Loading {cat} images from Drive"):
        images_list.append(load_all_seeds_for_prompt(cat, prompt, device))
        
    images = torch.stack(images_list)
    N = images.shape[0]
    lpip_matrix = np.zeros((N, N))

    with torch.no_grad():
        for i in tqdm(range(N), desc=f"Computing LPIPS pairs for {cat}"):
            for j in range(i+1, N):
                seed_distances = []
                
                for seed in range(5):
                    img_i = images[i, seed].unsqueeze(0)
                    img_j = images[j, seed].unsqueeze(0)
                    seed_distances.append(loss_fc(img_i, img_j).item())

                med_dist = np.median(seed_distances)
                lpip_matrix[i,j] = med_dist
                lpip_matrix[j,i] = med_dist

    cos_dists = 1.0 - cos_sims
    valid_pairs_mask = cos_dists > 1e-6
    ratios = np.zeros_like(lpip_matrix)
    ratios[valid_pairs_mask] = lpip_matrix[valid_pairs_mask] / cos_dists[valid_pairs_mask]
    
    lipschitz_constants[cat] = np.max(ratios)
    print(f">> Final {cat.upper()} Lipschitz Constant: {lipschitz_constants[cat]:.4f}\n")


