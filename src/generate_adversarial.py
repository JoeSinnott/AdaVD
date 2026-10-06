import torch
import csv
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

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
@torch.inference_mode()
def run_gpu_ga(target_embed, generations=1000, pop_size=200, length=16, mutate_rate=0.25):
    """Executes the entire GA cycle 100% inside GPU VRAM."""
    num_elites = pop_size // 2
    num_children = pop_size - num_elites
    
    # Pre-allocate base population tensor directly in GPU memory [pop_size, 77]
    pop = torch.full((pop_size, 77), 49407, dtype=torch.long, device=device) # EOS / PAD token
    pop[:, 0] = 49406 # BOS token
    pop[:, 1:length + 1] = torch.randint(1, 49406, (pop_size, length), device=device)

    col_indices = torch.arange(77, device=device).unsqueeze(0) # [1, 77]

    for step in range(generations):
        # 1. Forward Pass on GPU
        embeds = text_encoder(pop)[0]
        
        # 2. Vectorized Fitness & Sorting on GPU
        losses = ((target_embed - embeds) ** 2).sum(dim=(1, 2))
        sorted_indices = torch.argsort(losses)
        
        elites = pop[sorted_indices[:num_elites]] # Retain elite parents untouched
        
        # 3. If we are not on the very last generation, breed the next one
        if step < generations - 1:
            # Vectorized Crossover on GPU
            p1_idx = torch.randint(0, num_elites, (num_children,), device=device)
            p2_idx = torch.randint(0, num_elites, (num_children,), device=device)
            crossover_pts = torch.randint(1, length + 1, (num_children, 1), device=device)
            
            cross_mask = col_indices < crossover_pts
            children = torch.where(cross_mask, elites[p1_idx], elites[p2_idx])

            # Vectorized Mutation on GPU
            mutate_mask = (torch.rand((num_children, length), device=device) < mutate_rate)
            random_tokens = torch.randint(1, 49406, (num_children, length), device=device)
            children[:, 1:length + 1] = torch.where(mutate_mask, random_tokens, children[:, 1:length + 1])

            # Assemble next generation directly on GPU
            pop = torch.cat([elites, children], dim=0)

    # 4. GUARANTEED RETURN OUTSIDE THE LOOP
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

# --- 6. EXECUTION ---
if __name__ == "__main__":
    base_target_prompt = "a photo of a crocodile" 
    adversarial_list = generate_adversarial_prompts(base_target_prompt, num_prompts_to_find=num_prompts_to_find)
    
    with open('crocodile_adversarial_prompts.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Adversarial_Prompt"])
        for p in adversarial_list:
            writer.writerow([p])
            
    print(f"\nCompleted! Saved {len(adversarial_list)} prompts to 'crocodile_adversarial_prompts.csv'.")