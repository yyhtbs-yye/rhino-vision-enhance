#!/usr/bin/env python3
"""
Encode SDXL text conditioning for DF2K prompts.

Reads prompt text files from:
  data/df2k_download/datasets/DF2K/DF2K_train_prompt

Writes one safetensors file per prompt to:
  data/df2k_download/datasets/DF2K/DF2K_train_embeddings

Each output file stores:
  - prompt_embeds
  - pooled_prompt_embeds
  - add_time_ids

Default behavior:
  * Uses SDXL base model text encoders.
  * Uses the same prompt for both SDXL tokenizers/text encoders.

Example:
  pip install -U torch torchvision diffusers transformers accelerate safetensors pillow tqdm huggingface_hub
  python df2k_sdxl_embeddings.py --device cuda --dtype fp16
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel


DEFAULT_PROMPTS_DIR = Path("data/df2k_download/datasets/DF2K/DF2K_train_prompt")
DEFAULT_OUTPUT_DIR = Path("data/df2k_download/datasets/DF2K/DF2K_train_embeddings")
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode DF2K prompt txt files into SDXL embeddings.")
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-left", type=int, default=0)
    parser.add_argument("--target-height", type=int, default=None)
    parser.add_argument("--target-width", type=int, default=None)
    parser.add_argument(
        "--save-json-index",
        action="store_true",
        help="Also save a manifest.json listing all produced files and metadata.",
    )
    return parser.parse_args()


def pick_dtype(device: str, dtype_name: str | None) -> torch.dtype:
    if dtype_name is None:
        if device.startswith("cuda"):
            return torch.float16
        return torch.float32

    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = mapping[dtype_name]

    if device == "cpu" and dtype == torch.float16:
        warnings.warn("fp16 on CPU is usually slow/fragile; switching to fp32.")
        return torch.float32
    return dtype


def stem_to_int(stem: str) -> int | None:
    try:
        return int(stem)
    except ValueError:
        return None


def list_prompt_files(prompts_dir: Path, start_index: int | None, end_index: int | None) -> List[Path]:
    files = sorted(prompts_dir.glob("*.txt"))
    if start_index is None and end_index is None:
        return files

    selected: List[Path] = []
    for path in files:
        idx = stem_to_int(path.stem)
        if idx is None:
            continue
        if start_index is not None and idx < start_index:
            continue
        if end_index is not None and idx > end_index:
            continue
        selected.append(path)
    return selected


def batched(items: Sequence[Path], batch_size: int) -> Iterable[List[Path]]:
    for i in range(0, len(items), batch_size):
        yield list(items[i : i + batch_size])


def read_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


@torch.no_grad()
def load_sdxl_text_stack(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype,
    local_files_only: bool,
):
    tokenizer_1 = AutoTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
        use_fast=False,
        local_files_only=local_files_only,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer_2",
        use_fast=False,
        local_files_only=local_files_only,
    )

    text_encoder_1 = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    ).to(device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id,
        subfolder="text_encoder_2",
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    ).to(device)

    text_encoder_1.eval()
    text_encoder_2.eval()

    # Load only UNet config so we can validate add_time_ids layout without loading UNet weights.
    unet_config = UNet2DConditionModel.load_config(
        model_id,
        subfolder="unet",
        local_files_only=local_files_only,
    )

    return tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2, unet_config


@torch.no_grad()
def encode_prompt_batch(
    prompts: Sequence[str],
    tokenizer_1,
    tokenizer_2,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        prompt_embeds: [batch, 77, 2048]
        pooled_prompt_embeds: [batch, proj_dim]   # usually [batch, 1280] for SDXL base
    """
    prompt_embeds_list = []
    pooled_prompt_embeds = None

    encoders = (
        (tokenizer_1, text_encoder_1),
        (tokenizer_2, text_encoder_2),
    )

    for encoder_idx, (tokenizer, text_encoder) in enumerate(encoders):
        text_inputs = tokenizer(
            list(prompts),
            padding="max_length",
            max_length=tokenizer.model_max_length,   # SDXL => 77
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)

        # Optional truncation warning
        untruncated_ids = tokenizer(
            list(prompts),
            padding="longest",
            return_tensors="pt",
        ).input_ids
        if (
            untruncated_ids.shape[-1] >= text_input_ids.shape[-1]
            and not torch.equal(text_input_ids.cpu(), untruncated_ids)
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
            )
            warnings.warn(
                "Some prompts were truncated because the CLIP tokenizer hit its max length. "
                f"Truncated tail(s): {removed_text}"
            )

        outputs = text_encoder(
            text_input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        # SDXL prompt_embeds use the penultimate hidden state from each encoder.
        hidden_states = outputs.hidden_states[-2]
        prompt_embeds_list.append(hidden_states)

        # SDXL pooled_prompt_embeds should come from text encoder 2.
        if encoder_idx == 1:
            # CLIPTextModelWithProjection returns text_embeds as the projected pooled output.
            if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
                pooled_prompt_embeds = outputs.text_embeds
            else:
                # Fallback for older / variant behaviors
                pooled_prompt_embeds = outputs[0]

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)   # [B, 77, 2048]
    target_dtype = text_encoder_2.dtype

    prompt_embeds = prompt_embeds.to(device=device, dtype=target_dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=target_dtype)

    return prompt_embeds, pooled_prompt_embeds


def save_embedding_file(
    out_path: Path,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    prompt_text: str,
    model_id: str,
    source_prompt_path: Path,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tensors = {
        "prompt_embeds": prompt_embeds.detach().contiguous().cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.detach().contiguous().cpu(),
    }

    metadata = {
        "prompt": prompt_text,
        "model_id": model_id,
        "source_prompt_path": str(source_prompt_path),
        "tensor_format": "safetensors",
    }

    save_file(tensors, str(out_path), metadata=metadata)


def main() -> None:
    args = parse_args()

    if (args.target_height is None) ^ (args.target_width is None):
        raise ValueError("Please provide both --target-height and --target-width, or neither.")

    if not args.prompts_dir.exists():
        raise FileNotFoundError(f"Prompt directory not found: {args.prompts_dir}")

    prompt_files = list_prompt_files(args.prompts_dir, args.start_index, args.end_index)
    if not prompt_files:
        raise FileNotFoundError(f"No prompt txt files found in: {args.prompts_dir}")

    torch_dtype = pick_dtype(args.device, args.dtype)
    crop_top_left = (args.crop_top, args.crop_left)

    print(f"Found {len(prompt_files)} prompt file(s).")
    print(f"Loading SDXL text encoders from: {args.model_id}")
    print(f"Device: {args.device} | dtype: {torch_dtype}")

    tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2, unet_config = load_sdxl_text_stack(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=torch_dtype,
        local_files_only=args.local_files_only,
    )

    manifest = []

    to_process: List[Path] = []
    for prompt_path in prompt_files:
        out_path = args.output_dir / f"{prompt_path.stem}.safetensors"
        if out_path.exists() and not args.overwrite:
            continue
        to_process.append(prompt_path)

    print(f"Need to write {len(to_process)} embedding file(s).")
    if not to_process:
        print("Nothing to do.")
        return

    for batch_paths in tqdm(list(batched(to_process, args.batch_size)), desc="Encoding batches"):
        prompts = [read_prompt(p) for p in batch_paths]
        prompt_embeds_batch, pooled_prompt_embeds_batch = encode_prompt_batch(
            prompts=prompts,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            device=args.device,
        )

        for batch_idx, prompt_path in enumerate(batch_paths):
            stem = prompt_path.stem
            out_path = args.output_dir / f"{stem}.safetensors"
            prompt_text = prompts[batch_idx]

            save_embedding_file(
                out_path=out_path,
                prompt_embeds=prompt_embeds_batch[batch_idx],              # [77, 2048]
                pooled_prompt_embeds=pooled_prompt_embeds_batch[batch_idx],# [proj_dim]
                prompt_text=prompt_text,
                model_id=args.model_id,
                source_prompt_path=prompt_path,
            )
            
            manifest.append(
                {
                    "id": stem,
                    "prompt_path": str(prompt_path),
                    "embedding_path": str(out_path),
                }
            )

    if args.save_json_index:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        index_path = args.output_dir / "manifest.json"
        index_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Saved manifest: {index_path}")

    print("Done.")
    print(f"Embeddings written to: {args.output_dir}")


if __name__ == "__main__":
    main()
