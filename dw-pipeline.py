#!/usr/bin/env python3
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
import shutil


def load_config():
    """
    Load unified configuration from `configs/config.yaml`.

    The new config file combines dataset definitions and default settings.
    """
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def embed_all():
    """Embed the reference dataset and all candidates (if missing)."""
    config = load_config()
    artifacts_root = Path(config.get("artifacts", {}).get("root", "artifacts"))
    # Embed reference if needed
    ref = config["reference"]
    ref_meta = artifacts_root / "embeddings" / ref["name"] / "meta.json"
    if not ref_meta.exists():
        ref_cmd = [
            "dw-extract-embeddings",
            "--config", "configs/config.yaml",
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
        cand_meta = artifacts_root / "embeddings" / name / "meta.json"
        if not cand_meta.exists():
            cmd = [
                "dw-extract-embeddings",
                "--config", "configs/config.yaml",
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
    artifacts_root = Path(config.get("artifacts", {}).get("root", "artifacts"))
    ref = config["reference"]["name"]
    for name in config["candidates"]:
        cmd = [
            "dw-retrieve-1nn",
            "--ref_emb", str(artifacts_root / "embeddings" / ref / "emb.npy"),
            "--cand_emb", str(artifacts_root / "embeddings" / name / "emb.npy"),
            "--outdir", str(artifacts_root / "retrieval" / f"{ref}_{name}"),
        ]
        print(f"Retrieving {ref} → {name}...")
        subprocess.run(cmd, check=True)


def aggregate():
    """Aggregate per-query winners into dataset weights."""
    config = load_config()
    ref = config["reference"]["name"]
    datasets = list(config["candidates"].keys())
    artifacts_root = Path(config.get("artifacts", {}).get("root", "artifacts"))
    # One similarity vector per candidate; all must have the same length M (#queries).
    nn_sims = [str(artifacts_root / "retrieval" / f"{ref}_{d}" / "nn_sim.npy") for d in datasets]
    
    cmd = [
        "dw-aggregate-weights",
        "--datasets", *datasets,
        "--nn_sims", *nn_sims,
        "--outdir", str(artifacts_root / "weights" / ref),
    ]
    print(f"Aggregating weights for {len(datasets)} datasets...")
    subprocess.run(cmd, check=True)


def status():
    """Print which artifact groups exist (embeddings / retrieval / weights)."""
    config = load_config()
    ref = config["reference"]["name"]
    artifacts_root = Path(config.get("artifacts", {}).get("root", "artifacts"))
    
    print("=== Embeddings ===")
    for name in [ref] + list(config["candidates"].keys()):
        meta = artifacts_root / "embeddings" / name / "meta.json"
        if meta.exists():
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")
    
    print("\n=== Retrievals ===")
    for name in config["candidates"]:
        sim = artifacts_root / "retrieval" / f"{ref}_{name}" / "nn_sim.npy"
        print(f"{'✓' if sim.exists() else '✗'} {ref} → {name}")
    
    print("\n=== Weights ===")
    weights = artifacts_root / "weights" / ref / "weights.json"
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


def clean(dataset: str | None = None) -> None:
    """
    Delete artifacts for a specific dataset, or everything when no dataset is given.

    - clean <dataset>: remove embeddings + retrievals for that candidate
      (if <dataset> is the reference name, also remove its embeddings, retrievals, and weights).
    - clean: remove all embeddings/retrievals/weights after an interactive confirmation.
    """
    config = load_config()
    artifacts_root = Path(config.get("artifacts", {}).get("root", "artifacts"))
    ref = config["reference"]["name"]

    emb_root = artifacts_root / "embeddings"
    retr_root = artifacts_root / "retrieval"
    weights_root = artifacts_root / "weights"

    if dataset is None:
        ans = input(f"This will delete ALL embeddings/retrievals/weights under {artifacts_root}. Continue? [y/N]: ")
        if ans.lower() not in {"y", "yes"}:
            print("Aborted.")
            return
        for sub in (emb_root, retr_root, weights_root):
            if sub.exists():
                shutil.rmtree(sub)
        print("All artifacts removed.")
        return

    # Dataset-specific clean
    candidates = set(config.get("candidates", {}).keys())
    if dataset != ref and dataset not in candidates:
        print(f"Unknown dataset '{dataset}'. Known candidates: {sorted(candidates)}; reference: {ref}")
        sys.exit(1)

    if dataset == ref:
        # Reference: remove its embeddings, all retrievals involving it, and its weights.
        if (emb_root / ref).exists():
            shutil.rmtree(emb_root / ref)
        if retr_root.exists():
            for d in retr_root.glob(f"{ref}_*"):
                shutil.rmtree(d)
        if (weights_root / ref).exists():
            shutil.rmtree(weights_root / ref)
        print(f"Cleaned artifacts for reference dataset '{ref}'.")
        return

    # Candidate dataset: remove its embeddings and retrievals where it is the candidate.
    if (emb_root / dataset).exists():
        shutil.rmtree(emb_root / dataset)
    if (retr_root / f"{ref}_{dataset}").exists():
        shutil.rmtree(retr_root / f"{ref}_{dataset}")
    print(f"Cleaned artifacts for candidate dataset '{dataset}'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: dw-pipeline.py <command> [dataset]")
        print("Commands: embed, retrieve, pipeline, clean, embed-all, retrieve-all, aggregate, status")
        sys.exit(1)

    cmd = sys.argv[1]
    dataset = sys.argv[2] if len(sys.argv) >= 3 else None

    if cmd == "embed":
        if dataset is None:
            embed_all()
        else:
            # Embed only the specified dataset (reference or candidate).
            config = load_config()
            artifacts_root = Path(config.get("artifacts", {}).get("root", "artifacts"))
            ref = config["reference"]
            cands = config.get("candidates", {})

            if dataset == ref["name"]:
                print(f"Embedding reference {ref['name']} only...")
                ref_meta = artifacts_root / "embeddings" / ref["name"] / "meta.json"
                if not ref_meta.exists():
                    ref_cmd = [
                        "dw-extract-embeddings",
                        "--config", "configs/config.yaml",
                        "--dataset", ref["name"],
                        "--root", ref["root"],
                        "--pattern", ref["pattern"],
                    ]
                    if ref.get("max_frames"):
                        ref_cmd.extend(["--max_frames_per_scene", str(ref["max_frames"])])
                    if ref.get("batch_size"):
                        ref_cmd.extend(["--batch-size", str(ref["batch_size"])])
                    subprocess.run(ref_cmd, check=True)
                else:
                    print(f"Reference {ref['name']} already embedded")
            elif dataset in cands:
                spec = cands[dataset]
                cand_meta = artifacts_root / "embeddings" / dataset / "meta.json"
                if not cand_meta.exists():
                    cmd_args = [
                        "dw-extract-embeddings",
                        "--config", "configs/config.yaml",
                        "--dataset", dataset,
                        "--root", spec["root"],
                        "--pattern", spec["pattern"],
                    ]
                    if spec.get("max_frames"):
                        cmd_args.extend(["--max_frames_per_scene", str(spec["max_frames"])])
                    if spec.get("batch_size"):
                        cmd_args.extend(["--batch-size", str(spec["batch_size"])])
                    print(f"Embedding {dataset}...")
                    subprocess.run(cmd_args, check=True)
                else:
                    print(f"Candidate {dataset} already embedded")
            else:
                print(f"Unknown dataset '{dataset}'.")
                print(f"Reference: {ref['name']}")
                print(f"Candidates: {', '.join(sorted(cands.keys()))}")
                sys.exit(1)
    elif cmd == "retrieve":
        config = load_config()
        ref = config["reference"]["name"]
        artifacts_root = Path(config.get("artifacts", {}).get("root", "artifacts"))
        cands = config.get("candidates", {})

        if dataset is None:
            retrieve_all()
        else:
            if dataset not in cands:
                print(f"Unknown candidate dataset '{dataset}'. Candidates: {', '.join(sorted(cands.keys()))}")
                sys.exit(1)
            cmd_args = [
                "dw-retrieve-1nn",
                "--ref_emb", str(artifacts_root / "embeddings" / ref / "emb.npy"),
                "--cand_emb", str(artifacts_root / "embeddings" / dataset / "emb.npy"),
                "--outdir", str(artifacts_root / "retrieval" / f"{ref}_{dataset}"),
            ]
            print(f"Retrieving {ref} → {dataset}...")
            subprocess.run(cmd_args, check=True)
    elif cmd == "pipeline":
        config = load_config()
        if dataset is None:
            pipeline()
        else:
            # Per-dataset pipeline: embed + retrieve for that dataset, then aggregate over all.
            print(f"Running pipeline for dataset '{dataset}'...")
            artifacts_root = Path(config.get("artifacts", {}).get("root", "artifacts"))
            ref = config["reference"]
            cands = config.get("candidates", {})

            # Embed dataset (reuse logic from the 'embed' branch).
            if dataset == ref["name"]:
                print(f"Embedding reference {ref['name']} only...")
                ref_meta = artifacts_root / "embeddings" / ref["name"] / "meta.json"
                if not ref_meta.exists():
                    ref_cmd = [
                        "dw-extract-embeddings",
                        "--config", "configs/config.yaml",
                        "--dataset", ref["name"],
                        "--root", ref["root"],
                        "--pattern", ref["pattern"],
                    ]
                    if ref.get("max_frames"):
                        ref_cmd.extend(["--max_frames_per_scene", str(ref["max_frames"])])
                    if ref.get("batch_size"):
                        ref_cmd.extend(["--batch-size", str(ref["batch_size"])])
                    subprocess.run(ref_cmd, check=True)
                else:
                    print(f"Reference {ref['name']} already embedded")
            elif dataset in cands:
                spec = cands[dataset]
                cand_meta = artifacts_root / "embeddings" / dataset / "meta.json"
                if not cand_meta.exists():
                    cmd_args = [
                        "dw-extract-embeddings",
                        "--config", "configs/config.yaml",
                        "--dataset", dataset,
                        "--root", spec["root"],
                        "--pattern", spec["pattern"],
                    ]
                    if spec.get("max_frames"):
                        cmd_args.extend(["--max_frames_per_scene", str(spec["max_frames"])])
                    if spec.get("batch_size"):
                        cmd_args.extend(["--batch-size", str(spec["batch_size"])])
                    print(f"Embedding {dataset}...")
                    subprocess.run(cmd_args, check=True)
                else:
                    print(f"Candidate {dataset} already embedded")
            else:
                print(f"Unknown dataset '{dataset}'.")
                print(f"Reference: {ref['name']}")
                print(f"Candidates: {', '.join(sorted(cands.keys()))}")
                sys.exit(1)

            # If dataset is a candidate, run retrieval for it as well.
            if dataset in cands:
                ref_name = ref["name"]
                cmd_args = [
                    "dw-retrieve-1nn",
                    "--ref_emb", str(artifacts_root / "embeddings" / ref_name / "emb.npy"),
                    "--cand_emb", str(artifacts_root / "embeddings" / dataset / "emb.npy"),
                    "--outdir", str(artifacts_root / "retrieval" / f"{ref_name}_{dataset}"),
                ]
                print(f"Retrieving {ref_name} → {dataset}...")
                subprocess.run(cmd_args, check=True)

            # Aggregate over all candidates.
            aggregate()
    elif cmd == "clean":
        clean(dataset)
    elif cmd == "embed-all":
        embed_all()
    elif cmd == "retrieve-all":
        retrieve_all()
    elif cmd == "aggregate":
        aggregate()
    elif cmd == "status":
        status()
    else:
        print(f"Unknown command '{cmd}'")
        sys.exit(1)