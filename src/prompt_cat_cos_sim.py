from template import template_dict
import torch
from transformers import CLIPTextModel, CLIPTokenizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    target_concept = "a photo of a crocodile"
    device = "mps"

    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
                    "CompVis/stable-diffusion-v1-4", 
                    subfolder="text_encoder", 
                    torch_dtype=torch.float16
                ).to(device).eval()

    print(f"Encoding target: '{target_concept}'...")
    anchor_embed = compute_normalized_embeddings([target_concept], tokenizer, text_encoder, device)

    category_embeddings = {}
    for category, prompts in template_dict.items():
        print(f"Encoding {len(prompts)} prompts for category: '{category}'...")
        category_embeddings[category] = compute_normalized_embeddings(prompts, tokenizer, text_encoder, device)
        
    target_sim_data = []
    inter_prompt_matrices = {}

    for cat, embeds in category_embeddings.items():
        if cat in ["lexical", "multilingual", "contextual", "synonyms", "ring-a-bell-croc"]:
            # Distance from target
            sim_to_target = (embeds @ anchor_embed.T).squeeze().cpu().numpy()
            for sim in sim_to_target:
                target_sim_data.append({"Category": cat, "Cosine Similarity to Target": sim})
            
            # Distance between cat prompts
            inter_sim = (embeds @ embeds.T).cpu().numpy()
            inter_prompt_matrices[cat] = inter_sim

    df_sim = pd.DataFrame(target_sim_data)

    sns.set_theme(style="whitegrid")
    
    # Plot 1: target distance swarm
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_sim, x="Category", y="Cosine Similarity to Target", inner=None, color="lightgray")
    sns.swarmplot(data=df_sim, x="Category", y="Cosine Similarity to Target", size=4, palette="Dark2", hue="Category", legend=False)
    
    plt.title("Prompt Drift from Pure Target Concept ('a photo of a crocodile')", fontweight="bold")
    plt.tight_layout()
    plt.savefig("target_drift_swarm.png", dpi=300)
    plt.close()

    # Plot 2: inter prompt heatmaps
    fig, axes = plt.subplots(1, len(inter_prompt_matrices), figsize=(20, 4))
    
    for ax, (cat, matrix) in zip(axes, inter_prompt_matrices.items()):
        sns.heatmap(
            matrix, 
            ax=ax, 
            cmap="mako", 
            xticklabels=False, 
            yticklabels=False,
            vmin=0.0, vmax=1.0, 
            cbar=(cat == list(inter_prompt_matrices.keys())[-1])
        )
        ax.set_title(f"{cat.capitalize()}\nInter-Prompt", fontweight="bold")
        
    plt.tight_layout()
    plt.savefig("inter_prompt_heatmaps_split.png", dpi=300)
    plt.close()