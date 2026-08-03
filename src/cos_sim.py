import os
import sys
import argparse
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

from template import template_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Cosine Similarities between Target Concept and Prompt Templates"
    )
    parser.add_argument(
        '--sd_ckpt',
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Path or HuggingFace ID for Stable Diffusion checkpoint"
    )
    parser.add_argument(
        '--target_concept',
        type=str,
        default="crocodile",
        help="Target concept to compare against"
    )
    parser.add_argument(
        '--erase_type',
        type=str,
        default="contextual",
        help="Comma-separated template types from template.py (e.g. contextual,lexical,synonyms,multilingual)"
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default="cosine_similarities.txt",
        help="Path to write the text file output"
    )
    return parser.parse_args()


def get_pooled_embedding(text, tokenizer, text_encoder, device):
    """Extracts the CLIP embedding for a string."""
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = text_encoder(**inputs)
        
    return outputs.pooler_output


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading CLIP Text Encoder from '{args.sd_ckpt}' on {device}...")
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_ckpt, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_ckpt, subfolder="text_encoder").to(device)

    target_emb = get_pooled_embedding(args.target_concept, tokenizer, text_encoder, device)

    erase_types = [t.strip() for t in args.erase_type.split(',')]
    
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    out_lines = []
    header = f"Target Concept: {args.target_concept}\n\n"
    print(header)
    out_lines.append(header)

    for e_type in erase_types:
        if e_type not in template_dict:
            continue

        section_header = f"Category: {e_type}\n"
        print(section_header)
        out_lines.append(section_header)

        templates = template_dict[e_type]
        # format strings
        prompts = [t.format(args.target_concept) if "{}" in t else t for t in templates]

        cos_scores = []
        for prompt in prompts:
            prompt_emb = get_pooled_embedding(prompt, tokenizer, text_encoder, device)
            cos_sim = F.cosine_similarity(target_emb, prompt_emb, dim=-1).item()
            cos_scores.append(cos_sim)

            line = f"Score: {cos_sim:.4f} | Prompt: '{prompt}'\n"
            print(line.strip())
            out_lines.append(line)

        if cos_scores:
            avg_sim = sum(cos_scores) / len(cos_scores)
            summary_line = f"\nMean Cosine Similarity [{e_type}]: {avg_sim:.4f}\n\n"
            print(summary_line)
            out_lines.append(summary_line)

    with open(args.save_path, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)

    print(f"Results written to: '{args.save_path}'")


if __name__ == '__main__':
    main()