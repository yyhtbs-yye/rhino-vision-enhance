#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one text description per DF2K image with a LLaVA model "
                    "(multi-GPU parallel version)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/df2k_download/datasets/DF2K/DF2K_train_LR_bicubic/X4"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/df2k_download/datasets/DF2K/DF2K_train_prompt"),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="llava-hf/llava-v1.6-mistral-7b-hf",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Describe this image in one concise sentence for SDXL prompting, "
            "under 40 words. Focus on main subject, scene, lighting, and style."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use. Defaults to all available CUDA devices.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image discovery / filtering (unchanged from original)
# ---------------------------------------------------------------------------

def numeric_sort_key(path: Path):
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def list_images(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.png"), key=numeric_sort_key)


def filter_images(
    image_paths: Iterable[Path],
    start_index: int | None,
    end_index: int | None,
    limit: int | None,
) -> list[Path]:
    filtered: list[Path] = []
    for path in image_paths:
        if path.stem.isdigit():
            idx = int(path.stem)
            if start_index is not None and idx < start_index:
                continue
            if end_index is not None and idx > end_index:
                continue
        filtered.append(path)
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


# ---------------------------------------------------------------------------
# Per-GPU worker
# ---------------------------------------------------------------------------

def build_model_and_processor(model_id: str, gpu_id: int, args: argparse.Namespace):
    dtype = torch.float16

    model_kwargs: dict = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
        "device_map": {"": gpu_id},   # pin every layer to this specific GPU
    }

    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    return model, processor


def prepare_inputs(processor, image: Image.Image, prompt_text: str, device: str):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def decode_new_tokens(
    processor, generated_ids: torch.Tensor, prompt_length: int
) -> str:
    new_token_ids = generated_ids[:, prompt_length:]
    text = processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()
    for prefix in ("Assistant:", "ASSISTANT:", "assistant:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def worker(
    rank: int,
    image_shard: list[Path],
    args: argparse.Namespace,
    gpu_id: int,
) -> None:
    """Runs in a separate process; owns GPU `gpu_id` exclusively."""

    # Suppress duplicate log lines from child processes – only rank 0 logs progress.
    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format=f"GPU{gpu_id} | %(asctime)s | %(levelname)s | %(message)s",
    )

    device = f"cuda:{gpu_id}"
    model, processor = build_model_and_processor(args.model_id, gpu_id, args)

    processed = skipped = failed = 0

    for image_path in tqdm(
        image_shard,
        desc=f"GPU {gpu_id}",
        position=rank,
        leave=True,
    ):
        output_path = args.output_dir / f"{image_path.stem}.txt"

        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = prepare_inputs(processor, image, args.prompt, device)

            generation_kwargs: dict = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
            }
            if args.temperature > 0:
                generation_kwargs["temperature"] = args.temperature
                generation_kwargs["top_p"] = args.top_p

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **generation_kwargs)

            prompt_length = inputs["input_ids"].shape[1]
            caption = decode_new_tokens(processor, generated_ids, prompt_length)
            output_path.write_text(caption + "\n", encoding="utf-8")
            processed += 1

        except Exception as exc:  # noqa: BLE001
            failed += 1
            logging.exception("Failed on %s: %s", image_path.name, exc)

    logging.warning(
        "GPU %d done | processed=%d skipped=%d failed=%d",
        gpu_id,
        processed,
        skipped,
        failed,
    )


# ---------------------------------------------------------------------------
# Main – shard and spawn
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- GPU inventory -------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPUs found. This script requires at least one GPU.")

    total_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus or total_gpus, total_gpus)
    gpu_ids = list(range(num_gpus))
    logging.info("Using %d GPU(s): %s", num_gpus, gpu_ids)

    # ---- Image list ----------------------------------------------------------
    all_images = list_images(args.input_dir)
    image_paths = filter_images(
        all_images, args.start_index, args.end_index, args.limit
    )
    if not image_paths:
        raise RuntimeError(f"No PNG files found in {args.input_dir}")

    logging.info("Total images to process: %d", len(image_paths))

    # ---- Shard across GPUs (round-robin keeps numeric order within each shard)
    shards: list[list[Path]] = [[] for _ in gpu_ids]
    for i, path in enumerate(image_paths):
        shards[i % num_gpus].append(path)

    for i, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
        logging.info("GPU %d → %d images", gpu_id, len(shard))

    # ---- Spawn one process per GPU ------------------------------------------
    # 'spawn' is required on Linux when CUDA is involved to avoid fork-related
    # issues with CUDA contexts.
    mp.set_start_method("spawn", force=True)

    processes: list[mp.Process] = []
    for rank, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
        p = mp.Process(
            target=worker,
            args=(rank, shard, args, gpu_id),
            daemon=False,
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Surface any non-zero exit codes.
    for gpu_id, p in zip(gpu_ids, processes):
        if p.exitcode != 0:
            logging.error("Worker for GPU %d exited with code %d", gpu_id, p.exitcode)

    logging.info("All workers finished. Output: %s", args.output_dir)


if __name__ == "__main__":
    main()