# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert HuggingFace dataset format to WebDataset format.

This script keeps the numpy arrays in their original serialized bytes format and only
deserializes them during training/loading, which is more memory efficient.

WebDataset format:
- Each sample is stored with extensions: .json, .t5.bytes, .clip.bytes, .mean.bytes, .logvar.bytes
- The .bytes files contain the exact same serialized format as in the HuggingFace dataset
- Deserialization happens only when loading samples during training


Usage:
    requires an env with numpy, torch, hf_datasets and energon 7.1.0
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import tarfile

import numpy as np
import torch
from datasets import load_from_disk


def deserialize_numpy_array(data: bytes) -> np.ndarray:
    """Deserialize numpy array from bytes.

    Args:
        data: Serialized bytes
    """
    buffer = io.BytesIO(data)

    # Load uint16 view and convert back to bf16
    uint16_data = np.load(buffer)
    tensor = torch.from_numpy(uint16_data).view(torch.bfloat16)
    return tensor.numpy()


def process_shard(args):
    """Process a single shard in a separate process.

    Args:
        args: Tuple of (dataset_path, output_path, shard_idx, start_idx, end_idx, samples_per_shard)
    """
    dataset_path, output_path, shard_idx, start_idx, end_idx, samples_per_shard = args

    # Load dataset in this process (each process gets its own copy)
    dataset = load_from_disk(dataset_path)

    shard_path = os.path.join(output_path, f"shard_{shard_idx:06d}.tar")

    with tarfile.open(shard_path, "w") as tar:
        for i in range(start_idx, end_idx):
            sample = dataset[i]
            sample_key = sample["__key__"]

            # Keep the data in its original serialized format (bytes)
            # Do NOT deserialize - just pass the raw bytes through
            t5_bytes = sample["t5_encodings"]
            clip_bytes = sample["clip_encodings"]
            mean_bytes = sample["mean"]
            logvar_bytes = sample["logvar"]
            metadata = {"key": sample_key}

            if "timestep" in sample:
                timestep = sample["timestep"]
                metadata["timestep"] = timestep

            # Add files to tar with WebDataset naming convention
            # Use sample index as the base name for consistent ordering
            base_name = f"{i:08d}"

            # Add metadata as JSON
            metadata_bytes = json.dumps(metadata).encode("utf-8")
            metadata_info = tarfile.TarInfo(name=f"{base_name}.json")
            metadata_info.size = len(metadata_bytes)
            tar.addfile(metadata_info, io.BytesIO(metadata_bytes))

            # Add t5 encodings (already serialized bytes)
            t5_info = tarfile.TarInfo(name=f"{base_name}.t5.bytes")
            t5_info.size = len(t5_bytes)
            tar.addfile(t5_info, io.BytesIO(t5_bytes))

            # Add clip encodings (already serialized bytes)
            clip_info = tarfile.TarInfo(name=f"{base_name}.clip.bytes")
            clip_info.size = len(clip_bytes)
            tar.addfile(clip_info, io.BytesIO(clip_bytes))

            # Add mean (already serialized bytes)
            mean_info = tarfile.TarInfo(name=f"{base_name}.mean.bytes")
            mean_info.size = len(mean_bytes)
            tar.addfile(mean_info, io.BytesIO(mean_bytes))

            # Add logvar (already serialized bytes)
            logvar_info = tarfile.TarInfo(name=f"{base_name}.logvar.bytes")
            logvar_info.size = len(logvar_bytes)
            tar.addfile(logvar_info, io.BytesIO(logvar_bytes))

    return f"Completed shard {shard_idx+1}: {shard_path}"


def convert_to_webdataset(
    input_path: str,
    output_path: str,
    samples_per_shard: int = 10000,
    num_workers: int = 4,
):
    """Convert HuggingFace dataset to WebDataset format using multiprocessing.

    Args:
        input_path: Path to the HuggingFace dataset saved with save_to_disk
        output_path: Output directory for WebDataset shards
        samples_per_shard: Number of samples per tar shard
        num_workers: Number of worker processes for parallel shard processing
    """
    print(f"Loading dataset from {input_path}")
    dataset = load_from_disk(input_path)

    print(f"Dataset loaded with {len(dataset)} samples")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Calculate shard information
    num_shards = (len(dataset) + samples_per_shard - 1) // samples_per_shard
    print(f"Creating {num_shards} shards with up to {samples_per_shard} samples each")
    print(f"Using {num_workers} worker processes")

    # Prepare arguments for each shard
    shard_args = []
    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx = min(start_idx + samples_per_shard, len(dataset))
        shard_args.append(
            (input_path, output_path, shard_idx, start_idx, end_idx, samples_per_shard)
        )

    # Process shards in parallel
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(process_shard, shard_args)

    # Print results
    for result in results:
        print(result)

    print(f"Conversion complete! WebDataset saved to {output_path}")
    print(f"Created {num_shards} shards with file pattern: shard_XXXXXX.tar")
    print(
        f"Each sample contains: .json (metadata), .t5.bytes, .clip.bytes, .mean.bytes, .logvar.bytes"
    )


def create_index_file(output_path: str):
    """Create an index file listing all shards for easy loading.

    Args:
        output_path: Directory containing the WebDataset shards
    """
    shard_files = sorted(
        [
            f
            for f in os.listdir(output_path)
            if f.startswith("shard_") and f.endswith(".tar")
        ]
    )

    index_path = os.path.join(output_path, "index.txt")
    with open(index_path, "w") as f:
        for shard_file in shard_files:
            f.write(f"{shard_file}\n")

    print(f"Created index file: {index_path}")
    return len(shard_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace dataset to WebDataset format"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the HuggingFace dataset directory (output of preprocess_flux_dataset.py)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for WebDataset shards",
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=10000,
        help="Number of samples per WebDataset shard (default: 10000)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel shard processing (default: 4)",
    )

    args = parser.parse_args()

    # Validate input path
    if not os.path.exists(args.input_path):
        raise ValueError(f"Input path does not exist: {args.input_path}")

    # Set multiprocessing start method to 'spawn' for better compatibility
    if hasattr(mp, "set_start_method"):
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Start method already set

    # Convert to WebDataset
    convert_to_webdataset(
        args.input_path, args.output_path, args.samples_per_shard, args.num_workers
    )

    # Create index file
    num_shards = create_index_file(args.output_path)

    print("\nConversion Summary:")
    print(f"  Input: {args.input_path}")
    print(f"  Output: {args.output_path}")
    print(f"  Shards created: {num_shards}")
    print(f"  Samples per shard: {args.samples_per_shard}")
    print(f"  Worker processes: {args.num_workers}")


if __name__ == "__main__":
    main()
