"""
Dynamic pipeline runner.

This script orchestrates the CLI tools in `src/dw/cli/` to produce the standard
artifact layout under `artifacts/`:

- `artifacts/embeddings/<dataset>/...` (embedding extraction + metadata)
- `artifacts/retrieval/<ref>_<cand>/...` (1-NN retrieval results)
- `artifacts/weights/<ref>/...` (aggregated weights / wins)

It is intentionally *idempotent*: if `meta.json` already exists for a dataset,
we assume embeddings were produced previously and skip recomputation.

Usage: `python dw-pipeline.py <command>`
"""

import yaml
import subprocess
import sys
from pathlib import Path


def load_config():
    """Load `configs/datasets.yaml` describing reference + candidate datasets."""
    with open("configs/datasets.yaml") as f:
        return yaml.safe_load(f)


def embed_all():
    """Embed the reference dataset and all candidates (if missing)."""
    config = load_config()
    
    # Embed reference if needed
    ref = config["reference"]
    ref_meta = Path(f"artifacts/embeddings/{ref['name']}/meta.json")
    if not ref_meta.exists():
        ref_cmd = [
            "dw-extract-embeddings",
            "--config", "configs/default.yaml",
            "--dataset", ref["name"],
            "--root", ref["root"],
            "--pattern", ref["pattern"],
        ]
        # Optional knobs live in datasets.yaml (per dataset) and default.yaml (global).
        if ref.get("max_frames"):
            ref_cmd.extend(["--max_frames_per_scene", str(ref["max_frames"])])
        if ref.get("batch_size"):
            ref_cmd.extend(["--batch-size", str(ref["batch_size"])])
        print(f"Embedding reference {ref['name']}...")
        subprocess.run(ref_cmd, check=True)
    else:
        print(f"Reference {ref['name']} already embedded")
    
    # Embed candidates if needed
    for name, spec in config["candidates"].items():
        cand_meta = Path(f"artifacts/embeddings/{name}/meta.json")
        if not cand_meta.exists():
            cmd = [
                "dw-extract-embeddings",
                "--config", "configs/default.yaml",
                "--dataset", name,
                "--root", spec["root"],
                "--pattern", spec["pattern"],
            ]
            if spec.get("max_frames"):
                cmd.extend(["--max_frames_per_scene", str(spec["max_frames"])])
            # Per-dataset override (useful when mixing small/large images).
            if spec.get("batch_size"):
                cmd.extend(["--batch-size", str(spec["batch_size"])])
            print(f"Embedding {name}...")
            subprocess.run(cmd, check=True)
        else:
            print(f"Candidate {name} already embedded")


def retrieve_all():
    """Run 1-NN retrieval from reference → each candidate."""
    config = load_config()
    ref = config["reference"]["name"]
    for name in config["candidates"]:
        cmd = [
            "dw-retrieve-1nn",
            "--ref_emb", f"artifacts/embeddings/{ref}/emb.npy",
            "--cand_emb", f"artifacts/embeddings/{name}/emb.npy",
            "--outdir", f"artifacts/retrieval/{ref}_{name}",
        ]
        print(f"Retrieving {ref} → {name}...")
        subprocess.run(cmd, check=True)


def aggregate():
    """Aggregate per-query winners into dataset weights."""
    config = load_config()
    ref = config["reference"]["name"]
    datasets = list(config["candidates"].keys())
    # One similarity vector per candidate; all must have the same length M (#queries).
    nn_sims = [f"artifacts/retrieval/{ref}_{d}/nn_sim.npy" for d in datasets]
    
    cmd = [
        "dw-aggregate-weights",
        "--datasets", *datasets,
        "--nn_sims", *nn_sims,
        "--outdir", f"artifacts/weights/{ref}",
    ]
    print(f"Aggregating weights for {len(datasets)} datasets...")
    subprocess.run(cmd, check=True)


def status():
    """Print which artifact groups exist (embeddings / retrieval / weights)."""
    config = load_config()
    ref = config["reference"]["name"]
    
    print("=== Embeddings ===")
    for name in [ref] + list(config["candidates"].keys()):
        meta = Path(f"artifacts/embeddings/{name}/meta.json")
        if meta.exists():
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")
    
    print("\n=== Retrievals ===")
    for name in config["candidates"]:
        sim = Path(f"artifacts/retrieval/{ref}_{name}/nn_sim.npy")
        print(f"{'✓' if sim.exists() else '✗'} {ref} → {name}")
    
    print("\n=== Weights ===")
    weights = Path(f"artifacts/weights/{ref}/weights.json")
    if weights.exists():
        print(weights.read_text())
    else:
        print("Not computed")


def pipeline():
    """Full pipeline: embed → retrieve → aggregate."""
    embed_all()
    
    # Clear GPU memory before retrieval
    import torch
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    retrieve_all()  
    aggregate()

if __name__ == "__main__":
    commands = {
        "embed-all": embed_all,
        "retrieve-all": retrieve_all,
        "aggregate": aggregate,
        "status": status,
        "pipeline": pipeline,
    }
    
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage: python dw-pipeline.py <command>")
        print("Commands:", ", ".join(commands.keys()))
        sys.exit(1)
    
    commands[sys.argv[1]]()