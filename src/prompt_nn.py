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

    print("[DEBUG] Generating true Layer 12 contextualized vocabulary space...")
    vocab_size = len(tokenizer)
    start_id = tokenizer.bos_token_id
    end_id = tokenizer.eos_token_id
    
    batch_size = 2048
    layer12_vocab_list = []
    
    # Push the entire dictionary through CLIP to get Layer 12 representations
    with torch.no_grad():
        for i in range(0, vocab_size, batch_size):
            chunk_size = min(batch_size, vocab_size - i)
            tokens = torch.arange(i, i + chunk_size, device=device).unsqueeze(1)
            
            # Format: [BOS, TOKEN, EOS]
            bos_col = torch.full((chunk_size, 1), start_id, device=device)
            eos_col = torch.full((chunk_size, 1), end_id, device=device)
            batch_input_ids = torch.cat([bos_col, tokens, eos_col], dim=1)
            
            # Extract Layer 12 hidden states
            hidden_states = text_encoder(batch_input_ids).last_hidden_state
            
            # Grab the contextualized vector of the middle token (index 1)
            token_embeds = hidden_states[:, 1, :]
            layer12_vocab_list.append(token_embeds)
            
    vocab_layer12 = torch.cat(layer12_vocab_list, dim=0)
    vocab_normalized = torch.nn.functional.normalize(vocab_layer12, p=2, dim=1)

    print(f"\n[DEBUG] Target Adversarial Prompt: '{test_prompt}'")
    print("[DEBUG] Processing target prompt at Layer 12...")
    
    # Tokenize WITHOUT padding to prevent dilution
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    prompt_out = text_encoder(**inputs).last_hidden_state
    
    # Slice out BOS (0) and EOS (-1) to isolate the true contextualized words
    pure_word_embeds = prompt_out[0, 1:-1, :]
    
    # Average the contextualized words
    prompt_vector = pure_word_embeds.mean(dim=0).unsqueeze(0)
    prompt_normalized = torch.nn.functional.normalize(prompt_vector, p=2, dim=1)

    print("[DEBUG] Computing cosine similarities in Layer 12 space...")
    cos_sims = (prompt_normalized @ vocab_normalized.T).squeeze()
    
    top_10 = torch.topk(cos_sims, 10)
    top_10_indices = top_10.indices
    top_10_values = top_10.values
    top_10_tokens = tokenizer.convert_ids_to_tokens(top_10_indices.tolist())
    
    croc_token_id = tokenizer.encode("crocodile", add_special_tokens=False)[0]
    croc_sim = cos_sims[croc_token_id].item()
    
    clean_neighbors = [t.replace('</w>', '') for t in top_10_tokens]
    print(f"\n--- Top 10 Contextualized Neighbors Found ---")
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