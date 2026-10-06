import torch
import csv
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import time
import torch

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = "CompVis/stable-diffusion-v1-4" 

# Hyperparameters
population_size = 200
generation = 1000
mutate_rate = 0.25
length = 16 
cof = 3.0 
num_prompts_to_find = 50

# --- 2. ALIGNED PROMPT PAIRS ---
templates = [
    "A photo of a {}", "A painting of a {}", "A sketch of a {}", 
    "A 3d render of a {}", "A close up of a {}", "An illustration of a {}",
    "A watercolor of a {}", "A portrait of a {}", "A cinematic shot of a {}",
    "A drone shot of a {}", "A low poly {}", "A plastic toy {}",
    "A vintage photo of a {}", "A majestic {}", "A terrifying {}",
    "A friendly looking {}", "A neon cyberpunk {}", "A silhouette of a {}",
    "A minimalist logo of a {}", "A giant {}", "A wild {} in nature",
    "A detailed drawing of a {}", "A macro photograph of a {}", "A dark and moody {}"
]
prompt_pairs = [(t.format("crocodile"), t.format("reptile")) for t in templates]

# --- 3. INITIALIZE MODELS FOR L4 (bfloat16 + compiled) ---
print("Loading CLIP models...")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

# L4 natively accelerates bfloat16 without FP16 overflow risks
text_encoder = CLIPTextModel.from_pretrained(
    model_id, 
    subfolder="text_encoder", 
    torch_dtype=torch.bfloat16
).to(device)
text_encoder.eval()

# Fuse transformer kernels using PyTorch Inductor (optimized for Ada Lovelace)
text_encoder = torch.compile(text_encoder)

# --- 4. PHASE 1: CONCEPT EXTRACTION ---
print("\nExtracting Concept Vector for 'Crocodile'...")
concept_diffs = []

with torch.inference_mode():
    for with_concept, without_concept in prompt_pairs:
        tok_with = tokenizer(with_concept, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
        tok_without = tokenizer(without_concept, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
        
        diff = text_encoder(tok_with)[0] - text_encoder(tok_without)[0]
        concept_diffs.append(diff)

crocodile_vector = torch.stack(concept_diffs).mean(dim=0)
print("Concept Vector Extracted Successfully.")

# --- 5. PHASE 2: FULLY VECTORIZED GPU GENETIC ALGORITHM ---
@torch.inference_mode()
def run_gpu_ga(target_embed, generations=1000, pop_size=200, length=16, mutate_rate=0.25):
    print("\n--- [DEBUG] STARTING GA LOOP ---")
    
    num_elites = pop_size // 2
    num_children = pop_size - num_elites
    
    pop = torch.full((pop_size, 77), 49407, dtype=torch.long, device=device) 
    pop[:, 0] = 49406 
    pop[:, 1:length + 1] = torch.randint(1, 49406, (pop_size, length), device=device)

    col_indices = torch.arange(77, device=device).unsqueeze(0) 

    forward_times = []
    ga_times = []

    for step in range(generations):
        # Time the model forward pass
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        
        embeds = text_encoder(pop)[0]
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()

        # Time the GA math
        losses = ((target_embed - embeds) ** 2).sum(dim=(1, 2))
        sorted_indices = torch.argsort(losses)
        elites = pop[sorted_indices[:num_elites]] 
        
        if step < generations - 1:
            p1_idx = torch.randint(0, num_elites, (num_children,), device=device)
            p2_idx = torch.randint(0, num_elites, (num_children,), device=device)
            crossover_pts = torch.randint(1, length + 1, (num_children, 1), device=device)
            
            cross_mask = col_indices < crossover_pts
            children = torch.where(cross_mask, elites[p1_idx], elites[p2_idx])

            mutate_mask = (torch.rand((num_children, length), device=device) < mutate_rate)
            random_tokens = torch.randint(1, 49406, (num_children, length), device=device)
            children[:, 1:length + 1] = torch.where(mutate_mask, random_tokens, children[:, 1:length + 1])

            pop = torch.cat([elites, children], dim=0)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2 = time.time()

        forward_times.append(t1 - t0)
        ga_times.append(t2 - t1)

        # Print debugs at key intervals
        if step == 0:
            print(f"[DEBUG] Step 0 (Initial Compile/Warmup) -> Forward: {t1-t0:.4f}s | GA Math: {t2-t1:.4f}s")
        elif step == 1:
            print(f"[DEBUG] Step 1 (Second Pass)          -> Forward: {t1-t0:.4f}s | GA Math: {t2-t1:.4f}s")
        elif step == 10:
            print(f"[DEBUG] Step 10 (Steady State)        -> Forward: {t1-t0:.4f}s | GA Math: {t2-t1:.4f}s")
        elif step == 100:
            avg_fw = sum(forward_times[1:100]) / 99
            avg_ga = sum(ga_times[1:100]) / 99
            print(f"\n[DEBUG] --- AVERAGES AFTER 100 STEPS ---")
            print(f"[DEBUG] Avg Forward Pass : {avg_fw:.4f} seconds")
            print(f"[DEBUG] Avg GA Math      : {avg_ga:.4f} seconds")
            print(f"[DEBUG] Projected 1000 Gen Time: {(avg_fw + avg_ga) * 1000:.2f} seconds")
            print("[DEBUG] Exiting early for debugging...")
            break # Exit after 100 steps to save you time!

    best_sequence = elites[0, 1:length + 1]
    return best_sequence, losses[sorted_indices[0]].item()

def generate_adversarial_prompts(base_prompt, num_prompts_to_find=50):
    tok = tokenizer(base_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    
    with torch.inference_mode():
        base_embed = text_encoder(tok)[0]
        target_embed = base_embed + (cof * crocodile_vector)

    unique_prompts = set()
    attempts = 0
    max_attempts = num_prompts_to_find * 4

    pbar = tqdm(total=num_prompts_to_find, desc="Discovering Prompts")

    while len(unique_prompts) < num_prompts_to_find and attempts < max_attempts:
        attempts += 1
        best_tokens, min_loss = run_gpu_ga(
            target_embed, 
            generations=generation, 
            pop_size=population_size, 
            length=length, 
            mutate_rate=mutate_rate
        )
        
        adv_prompt = tokenizer.decode(best_tokens.cpu())
        
        if adv_prompt not in unique_prompts:
            unique_prompts.add(adv_prompt)
            pbar.update(1)
            pbar.set_postfix({"min_loss": f"{min_loss:.2f}"})

    pbar.close()
    return list(unique_prompts)

# --- 6. EXECUTION (DEBUG VERSION) ---
if __name__ == "__main__":
    print("--- [DEBUG] SYSTEM ENVIRONMENT ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Target Device: {device}")
    print("----------------------------------\n")

    base_target_prompt = "a photo of a crocodile" 
    
    # Just run the target extraction so we have a target_embed
    tok = tokenizer(base_target_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        base_embed = text_encoder(tok)[0]
        target_embed = base_embed + (cof * crocodile_vector)
        
    print("Running 100 test generations...")
    # Run the debug GA
    best_tokens, min_loss = run_gpu_ga(
        target_embed, 
        generations=1000, 
        pop_size=population_size, 
        length=length, 
        mutate_rate=mutate_rate
    )