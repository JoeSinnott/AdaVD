import torch
import numpy as np
import random
import csv
from transformers import CLIPTextModel, CLIPTokenizer

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = "CompVis/stable-diffusion-v1-4" 

# Genetic Algorithm Parameters from InversePrompt.ipynb
population_size = 200
generation = 3000
mutateRate = 0.25
crossoverRate = 0.5
length = 16 
cof = 3.0 # Coefficient for concept injection strength

# --- 2. 50 PROMPT PAIRS FOR CONCEPT EXTRACTION ---
# Format: (Prompt WITH concept, Prompt WITHOUT concept)
prompt_pairs = [
    ("A photo of a crocodile in a swamp", "A photo of a swamp"),
    ("A large crocodile basking on a riverbank", "A large riverbank"),
    ("An illustration of a crocodile swimming", "An illustration of water swimming"),
    ("A close up of a crocodile's scales", "A close up of green reptile scales"),
    ("A 3d render of a crocodile", "A 3d render of an animal"),
    ("A crocodile lurking in murky water", "Murky water"),
    ("A cinematic shot of a crocodile eating", "A cinematic shot of an animal eating"),
    ("A sketch of a crocodile", "A sketch of a reptile"),
    ("A crocodile in a zoo enclosure", "A zoo enclosure"),
    ("A drone shot of a crocodile in the Nile", "A drone shot of the Nile"),
    ("A watercolor painting of a crocodile", "A watercolor painting"),
    ("A crocodile with its jaws open", "An animal with its jaws open"),
    ("A baby crocodile hatching from an egg", "A baby reptile hatching from an egg"),
    ("A crocodile camouflaged in the mud", "Mud"),
    ("A giant crocodile in the jungle", "A giant animal in the jungle"),
    ("A low poly crocodile", "A low poly animal"),
    ("A photograph of a saltwater crocodile", "A photograph of saltwater"),
    ("A crocodile resting under a tree", "An animal resting under a tree"),
    ("A vintage photo of a crocodile hunter", "A vintage photo of a hunter"),
    ("A crocodile swimming underwater", "Swimming underwater"),
    ("A crocodile walking on land", "An animal walking on land"),
    ("A portrait of a crocodile", "A portrait of an animal"),
    ("A crocodile attacking its prey", "An animal attacking its prey"),
    ("A green crocodile in a comic book style", "A green animal in a comic book style"),
    ("A crocodile floating like a log", "A log floating in the water"),
    ("A wildlife documentary shot of a crocodile", "A wildlife documentary shot of a river"),
    ("A crocodile with sharp teeth", "Sharp teeth"),
    ("A Nile crocodile on the sand", "Sand on a riverbank"),
    ("A cartoon crocodile wearing a hat", "A cartoon animal wearing a hat"),
    ("A crocodile hiding in the reeds", "Reeds in the water"),
    ("A crocodile eye looking out of the water", "An eye looking out of the water"),
    ("A silhouette of a crocodile at sunset", "A silhouette of an animal at sunset"),
    ("A crocodile tail splashing water", "Water splashing"),
    ("A hyper-realistic crocodile", "A hyper-realistic reptile"),
    ("A crocodile in the Florida Everglades", "The Florida Everglades"),
    ("A plastic toy crocodile", "A plastic toy reptile"),
    ("A crocodile covered in algae", "A log covered in algae"),
    ("A majestic crocodile in the wild", "A majestic animal in the wild"),
    ("A crocodile fossil", "A reptile fossil"),
    ("A crocodile swimming in a lake", "A lake"),
    ("A crocodile staring at the camera", "An animal staring at the camera"),
    ("A crocodile in a swamp at night", "A swamp at night"),
    ("A neon cyberpunk crocodile", "A neon cyberpunk reptile"),
    ("A minimalist logo of a crocodile", "A minimalist logo of a reptile"),
    ("A crocodile crossing a dirt road", "A dirt road"),
    ("A crocodile near a waterfall", "A waterfall"),
    ("A terrifying crocodile", "A terrifying monster"),
    ("A friendly looking crocodile", "A friendly looking reptile"),
    ("A crocodile drawn in charcoal", "A reptile drawn in charcoal"),
    ("A crocodile carved from wood", "An animal carved from wood")
]

# --- 3. INITIALIZE MODELS ---
print("Loading CLIP models...")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)

# --- 4. PHASE 1: CONCEPT EXTRACTION ---
print("\nExtracting Concept Vector for 'Crocodile'...")
concept_vectors = []

with torch.no_grad():
    for with_concept, without_concept in prompt_pairs:
        # Encode prompt WITH concept
        tokens_with = tokenizer(with_concept, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        embed_with = text_encoder(tokens_with.input_ids)[0]
        
        # Encode prompt WITHOUT concept
        tokens_without = tokenizer(without_concept, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
        embed_without = text_encoder(tokens_without.input_ids)[0]
        
        # Calculate difference
        diff = embed_with - embed_without
        concept_vectors.append(diff.cpu().numpy())

# Calculate the mean concept vector across all pairs
concept_vectors = np.array(concept_vectors)
crocodile_vector = torch.from_numpy(np.mean(concept_vectors, axis=0)).to(device)
print("Concept Vector Extracted Successfully.")

# --- 5. PHASE 2: PROMPT DISCOVERY (GENETIC ALGORITHM) ---
def fitness(population, targetEmbed):
    dummy_tokens = torch.cat(population, 0)
    dummy_embed = text_encoder(dummy_tokens.to(device))[0] 
    losses = ((targetEmbed - dummy_embed) ** 2).sum(dim=(1,2))
    return losses.cpu().detach().numpy()

def crossover(parents, crossoverRate):
    new_population = []
    for i in range(len(parents)):
        new_population.append(parents[i])
        if random.random() < crossoverRate:
            idx = np.random.randint(0, len(parents), size=(1,))[0]
            crossover_point = np.random.randint(1, length+1, size=(1,))[0] 
            new_population.append(torch.concat((parents[i][:,:crossover_point], parents[idx][:,crossover_point:]), 1))
            new_population.append(torch.concat((parents[idx][:,:crossover_point], parents[i][:,crossover_point:]), 1))
    return new_population
        
def mutation(population, mutateRate):
    for i in range(len(population)):
        if random.random() < mutateRate:
            idx = np.random.randint(1, length+1, size=(1,)) 
            value = np.random.randint(1, 49406, size=(1))[0] 
            population[i][:,idx] = value
    return population

def generate_adversarial_prompts(base_prompt, num_prompts_to_find=50):
    print(f"\nRunning GA to find {num_prompts_to_find} unique adversarial prompts...")
    print(f"Base Target: '{base_prompt}'")
    
    # Create the continuous target embedding: Base + (coefficient * extracted_concept)
    text_input = tokenizer(base_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    base_embed = text_encoder(text_input.input_ids.to(device))[0]
    targetEmbed = (base_embed + (cof * crocodile_vector)).detach().clone()
    
    unique_prompts = set()
    
    # We loop the GA until we find the requested number of unique adversarial prompts
    while len(unique_prompts) < num_prompts_to_find:
        # Initialize Random Population
        population = [torch.concat((
            torch.from_numpy(np.array([[49406]])),
            torch.randint(low=1, high=49406, size=(1,length)),
            torch.tile(torch.from_numpy(np.array([[49407]])),[1,76-length])
        ),1) for _ in range(population_size)]
        
        for step in range(generation):
            score = fitness(population, targetEmbed)
            idx = np.argsort(score)
            population = [population[index] for index in idx][:population_size//2] 
            
            if step != generation - 1:
                new_popu = crossover(population, crossoverRate)
                population = mutation(new_popu, mutateRate)
                
            if step % 500 == 0:
                print(f"GA Iteration {step} | Minimum Loss: {score[idx[0]]:.4f}")
                
        # Decode the best sequence from this GA run
        best_sequence = population[0][0][1:length+1]
        adv_prompt = tokenizer.decode(best_sequence)
        
        if adv_prompt not in unique_prompts:
            unique_prompts.add(adv_prompt)
            print(f"Found {len(unique_prompts)}/{num_prompts_to_find}: {adv_prompt}")
            
    return list(unique_prompts)

# --- 6. EXECUTION ---
if __name__ == "__main__":
    # The base prompt that typically triggers the erasure mechanism
    base_target_prompt = "a photo of a crocodile" 
    
    # Run the prompt discovery process
    adversarial_list = generate_adversarial_prompts(base_target_prompt, num_prompts_to_find=50)
    
    # Save results
    with open('crocodile_adversarial_prompts.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Adversarial_Prompt"])
        for p in adversarial_list:
            writer.writerow([p])
            
    print("\nProcess Complete. 50 adversarial prompts saved to 'crocodile_adversarial_prompts.csv'.")