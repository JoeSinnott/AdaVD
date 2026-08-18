import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


class FastCLIPEvaluator:

    def __init__(
        self,
        model_name="openai/clip-vit-large-patch14",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

    @torch.no_grad()
    def score_batch(self, image_paths, text_prompts):
        images = [Image.open(p).convert("RGB") for p in image_paths]

        # Vision features
        img_inputs = self.processor(images=images, return_tensors="pt").to(
            self.device
        )
        img_feats = self.model.get_image_features(**img_inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        # Text features (Target concept or Prompt)
        text_inputs = self.tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        text_feats = self.model.get_text_features(**text_inputs)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        scores = (img_feats * text_feats).sum(dim=-1).detach().cpu().numpy()
        return scores * 100.0  # Scale to standard [0, 100]


def calculate_category_fid(
    orig_dir, retain_dir, device="cuda", batch_size=32, dims=2048
):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to(device)

    m1, s1 = fid_score.compute_statistics_of_path(
        orig_dir, inception_model, batch_size, dims, device, num_workers=2
    )
    m2, s2 = fid_score.compute_statistics_of_path(
        retain_dir, inception_model, batch_size, dims, device, num_workers=2
    )
    return fid_score.calculate_frechet_distance(m1, s1, m2, s2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_root",
        type=str,
        default="/content/drive/MyDrive/AdaVDExperiments",
    )
    parser.add_argument("--target_concept", type=str, default="crocodile")
    parser.add_argument(
        "--erase_types",
        type=str,
        default="lexical,contextual,multilingual,synonyms",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_eval = FastCLIPEvaluator(device=device)

    erase_types = [e.strip() for e in args.erase_types.split(",")]
    concept_dir_name = args.target_concept.replace(", ", "_")

    records = []
    category_fid_map = {}

    print(f"\n--- Starting Evaluation for Target: '{args.target_concept}' ---")

    for e_type in erase_types:
        base_dir = os.path.join(
            args.save_root, concept_dir_name, e_type, args.target_concept
        )
        orig_dir = os.path.join(base_dir, "original")
        retain_dir = os.path.join(base_dir, "retain")

        if not os.path.exists(orig_dir) or not os.path.exists(retain_dir):
            print(f"[Warning] Skipping {e_type}: Directory not found ({base_dir})")
            continue

        # 1. Calculate Category-level FID
        print(f"\nComputing FID for category: {e_type}...")
        fid_val = calculate_category_fid(
            orig_dir, retain_dir, device=device, batch_size=args.batch_size
        )
        category_fid_map[e_type] = fid_val
        print(f"[{e_type}] FID (Original vs Retain): {fid_val:.4f}")

        # 2. Match images and calculate Image-Level CLIP
        orig_images = sorted(glob.glob(os.path.join(orig_dir, "*.png")))
        retain_images = sorted(glob.glob(os.path.join(retain_dir, "*.png")))

        print(f"Computing CLIP scores for {len(retain_images)} image pairs...")
        for i in range(0, len(retain_images), args.batch_size):
            b_orig_paths = orig_images[i : i + args.batch_size]
            b_ret_paths = retain_images[i : i + args.batch_size]

            # Reconstruct prompt name from filename (removes the _0.png index)
            b_prompts = [
                re.sub(
                    r"_\d+\.png$",
                    "",
                    os.path.basename(p).replace("_", " ").strip(),
                )
                for p in b_ret_paths
            ]

            # Measure alignment against target concept ("crocodile")
            target_prompts = [
                f"a photo of a {args.target_concept}"
            ] * len(b_ret_paths)

            orig_clip = clip_eval.score_batch(b_orig_paths, target_prompts)
            ret_clip = clip_eval.score_batch(b_ret_paths, target_prompts)

            for orig_p, ret_p, prompt_str, c_orig, c_ret in zip(
                b_orig_paths, b_ret_paths, b_prompts, orig_clip, ret_clip
            ):
                records.append(
                    {
                        "category": e_type,
                        "prompt": prompt_str,
                        "image_name": os.path.basename(ret_p),
                        "clip_original": c_orig,
                        "clip_retain": c_ret,
                        "clip_delta": c_orig - c_ret,
                        "fid": fid_val,
                    }
                )

    df = pd.DataFrame(records)
    out_csv = os.path.join(
        args.save_root, concept_dir_name, "metrics_summary.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"\n[Done] Successfully saved full quantitative metrics to:\n{out_csv}")


if __name__ == "__main__":
    main()