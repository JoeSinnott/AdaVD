import os
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from template import template_dict 

@torch.inference_mode()
def test_clip_neighborhood():
    device = "cuda"
    save_dir = "/content/drive/MyDrive/AdaVD Images and Figures/Exp1_sb_0.7/crocodile/clip_neighbors/test_run"
    os.makedirs(save_dir, exist_ok=True)
    print(f"[DEBUG] Save directory ready: {save_dir}")

    print("[DEBUG] Loading Stable Diffusion pipeline (includes CLIP tokenizer & text encoder)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", 
        torch_dtype=torch.float16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    print("[DEBUG] Models loaded successfully to GPU.")

    print("[DEBUG] Extracting CLIP vocabulary weights...")
    vocab_embeddings = text_encoder.get_input_embeddings().weight.float()
    print(f"[DEBUG] Raw vocab_embeddings shape: {vocab_embeddings.shape} (Expected: [49408, 768])")
    vocab_normalized = torch.nn.functional.normalize(vocab_embeddings, p=2, dim=1)

    # just one contextual prompt for this test
    test_prompt = template_dict['contextual'][0]
    print(f"\n[DEBUG] Target Adversarial Prompt: '{test_prompt}'")

    print("[DEBUG] Tokenizing and embedding target prompt...")
    inputs = tokenizer(
        test_prompt, 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    ).to(device)
    
    prompt_embed = text_encoder(inputs.input_ids)[0]
    print(f"[DEBUG] Initial prompt embedding shape: {prompt_embed.shape} (Expected: [1, 77, 768])")
    
    prompt_vector = prompt_embed.float().mean(dim=1) 
    prompt_normalized = torch.nn.functional.normalize(prompt_vector, p=2, dim=1)
    print(f"[DEBUG] Final normalized prompt vector shape: {prompt_normalized.shape} (Expected: [1, 768])")

    print("[DEBUG] Computing cosine similarities against all vocabulary tokens...")
    cos_sims = (prompt_normalized @ vocab_normalized.T).squeeze()
    
    top_10 = torch.topk(cos_sims, 10)
    top_10_indices = top_10.indices
    top_10_values = top_10.values
    top_10_tokens = tokenizer.convert_ids_to_tokens(top_10_indices.tolist())
    
    croc_token_id = tokenizer("crocodile", add_special_tokens=False).input_ids[0]
    croc_sim = cos_sims[croc_token_id].item()
    
    clean_neighbors = [t.replace('</w>', '') for t in top_10_tokens]
    print(f"\n--- Top 10 Mathematical Neighbors Found ---")
    print(f"BASELINE: 'crocodile' (Index: {croc_token_id}) -> Cosine Sim: {croc_sim:.4f}")
    print("-" * 43)
    for i, word in enumerate(clean_neighbors):
        print(f"{i+1}. {word} (Index: {top_10_indices[i].item()}) -> Cosine Sim: {top_10_values[i].item():.4f}")
    print("-------------------------------------------\n")

    print("[DEBUG] Beginning image generation loop...")
    for i, neighbor_word in enumerate(clean_neighbors):
        new_prompt = f"a photo of a {neighbor_word}"
        print(f"[DEBUG] ({i+1}/10) Generating 5 seeds for prompt: '{new_prompt}'")
        
        generators = [torch.Generator(device=device).manual_seed(s) for s in range(5)]
        
        images = pipe(
            prompt=[new_prompt] * 5, 
            num_inference_steps=50, 
            generator=generators
        ).images
        
        for idx, img in enumerate(images):
            # fallback for weird tokens to prevent file saving errors
            safe_name = "".join([c for c in neighbor_word if c.isalnum()])
            if not safe_name: 
                safe_name = f"token_{top_10_indices[i].item()}"
                
            save_path = f"{save_dir}/{safe_name}_seed_{idx}.png"
            img.save(save_path)
            
        print(f"[DEBUG] Saved 5 images for '{neighbor_word}'.")
            
    print(f"\n[DEBUG] Done! Test batch saved to {save_dir}")

if __name__ == '__main__':
    test_clip_neighborhood()