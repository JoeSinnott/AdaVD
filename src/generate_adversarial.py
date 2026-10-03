import torch
import numpy as np
import random
from transformers import CLIPTextModel, CLIPTokenizer

# Parameters of Gene-Algo adapted from Ring-A-Bell
population_size = 200
generation = 1000  # Reduced from 3000 for faster iteration
mutateRate = 0.25
crossoverRate = 0.5
length = 16 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dir_ = "CompVis/stable-diffusion-v1-4" 
tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder").to(device)

def fitness(population, targetEmbed):
    # L2 distance between dummy token embeddings and target concept
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
            # Crossover avoiding the start token at index 0
            crossover_point = np.random.randint(1, length+1, size=(1,))[0] 
            new_population.append(torch.concat((parents[i][:,:crossover_point],parents[idx][:,crossover_point:]), 1))
            new_population.append(torch.concat((parents[idx][:,:crossover_point],parents[i][:,crossover_point:]), 1))
    return new_population
        
def mutation(population, mutateRate):
    for i in range(len(population)):
        if random.random() < mutateRate:
            idx = np.random.randint(1, length+1, size=(1,)) 
            # Choose meaningful tokens (Avoid 0, 49406, 49407)
            value = np.random.randint(1, 49406, size=(1))[0] 
            population[i][:,idx] = value
    return population

def generate_adversarial_prompts(target_concept, num_prompts=50):
    text_input = tokenizer(
        target_concept, 
        padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, 
        return_tensors="pt"
    )
    # Direct targeting instead of adding a conceptual direction vector
    targetEmbed = text_encoder(text_input.input_ids.to(device))[0].detach().clone()
    
    unique_prompts = set()
    print(f"Generating adversarial prompts for: '{target_concept}'")
    
    while len(unique_prompts) < num_prompts:
        # Initialize random population bounded by start and end tokens
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
                
        # Decode the best sequence (indices 1 to length)
        best_sequence = population[0][0][1:length+1]
        adv_prompt = tokenizer.decode(best_sequence)
        
        if adv_prompt not in unique_prompts:
            unique_prompts.add(adv_prompt)
            print(f"Found {len(unique_prompts)}/{num_prompts}: {adv_prompt}")

    return list(unique_prompts)

if __name__ == "__main__":
    adversarial_list = generate_adversarial_prompts("a photo of a crocodile", num_prompts=50)
    
    print("\n--- Copy and paste this into template.py ---")
    print("adversarial_templates = [")
    for p in adversarial_list:
        print(f'    "{p}",')
    print("]")