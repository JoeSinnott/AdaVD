import torch
import random
import csv
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = "CompVis/stable-diffusion-v1-4" 

population_size = 200
generation = 1000
mutate_rate = 0.25
base_length = 16 
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

# --- 3. INITIALIZE MODELS FOR L4 ---
print("Loading CLIP models...")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    model_id, 
    subfolder="text_encoder", 
    torch_dtype=torch.bfloat16
).to(device)
text_encoder.eval()
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

# --- 5. PHASE 2: VECTORIZED GPU GA ---
@torch.inference_mode()
def run_gpu_ga(target_embed, generations, pop_size, length, mutate_rate):
    num_elites = pop_size // 2
    num_children = pop_size - num_elites
    
    pop = torch.full((pop_size, 77), 49407, dtype=torch.long, device=device) 
    pop[:, 0] = 49406 
    pop[:, 1:length + 1] = torch.randint(1, 49406, (pop_size, length), device=device)
    col_indices = torch.arange(77, device=device).unsqueeze(0) 

    for step in range(generations):
        embeds = text_encoder(pop)[0]
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

    best_sequence = elites[0, 1:length + 1]
    return best_sequence, losses[sorted_indices[0]].item()

def generate_adversarial_prompts(base_prompt, num_prompts_to_find=50):
    tok = tokenizer(base_prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to(device)
    
    with torch.inference_mode():
        base_embed = text_encoder(tok)[0]
        base_target = base_embed + (cof * crocodile_vector)
        target_std = base_target.std().item() # Get standard deviation for noise scaling

    unique_prompts = set()
    attempts = 0
    max_attempts = num_prompts_to_find * 4 

    pbar = tqdm(total=num_prompts_to_find, desc="Discovering Prompts")

    while len(unique_prompts) < num_prompts_to_find and attempts < max_attempts:
        attempts += 1
        
        # --- THE FIX: JITTER THE SEARCH SPACE ---
        # 1. Randomize token length slightly (e.g., 14 to 18 tokens)
        current_length = random.randint(base_length - 2, base_length + 2)
        
        # 2. Add 5% noise to the embedding target to force the GA into a different path
        noise = torch.randn_like(base_target) * (target_std * 0.05)
        jittered_target = base_target + noise
        
        # Run the GA with the slightly altered target and length
        best_tokens, min_loss = run_gpu_ga(
            jittered_target, 
            generations=generation, 
            pop_size=population_size, 
            length=current_length, 
            mutate_rate=mutate_rate
        )
        
        adv_prompt = tokenizer.decode(best_tokens.cpu())
        
        if adv_prompt not in unique_prompts:
            unique_prompts.add(adv_prompt)
            pbar.update(1)
        
        # Update progress bar so you can see if it's hitting duplicates
        dupes = attempts - len(unique_prompts)
        pbar.set_postfix({"loss": f"{min_loss:.2f}", "dupes_hit": dupes})

    pbar.close()
    return list(unique_prompts)

# --- 6. EXECUTION ---
if __name__ == "__main__":
    base_target_prompt = "a photo of a crocodile" 
    
    print("\nStarting Adversarial Prompt Discovery...")
    adversarial_list = generate_adversarial_prompts(base_target_prompt, num_prompts_to_find=num_prompts_to_find)
    
    with open('crocodile_adversarial_prompts.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Adversarial_Prompt"])
        for p in adversarial_list:
            writer.writerow([p])
            
    print(f"\nCompleted! Saved {len(adversarial_list)} prompts to 'crocodile_adversarial_prompts.csv'.")