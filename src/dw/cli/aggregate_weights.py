"""
Aggregate dataset weights from per-dataset similarity vectors.

Given K similarity arrays (one per candidate) over the same M reference queries,
we assign each query to the candidate with the highest similarity ("win").
Weights are computed as win frequency (wins / M).
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--nn_sims", nargs="+", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    # Auto-detect if nn_sims not provided
    if args.nn_sims is None:
        args.nn_sims = []
        for dataset in args.datasets:
            sim_path = Path("artifacts/retrieval") / dataset / "nn_sim.npy"
            if not sim_path.exists():
                raise FileNotFoundError(
                    f"File not found: {sim_path}\n"
                    "If file name is different, provide with --nn_sims argument."
                )
            args.nn_sims.append(str(sim_path))

    K = len(args.datasets)
    if len(args.nn_sims) != K:
        raise ValueError("datasets and nn_sims must have same length")

    sims = [np.load(p).astype(np.float32) for p in args.nn_sims]
    M = sims[0].shape[0]
    for s in sims:
        assert s.shape == (M,)

    S = np.stack(sims, axis=0)

    # For each query, pick the best matching dataset index.
    wins = np.argmax(S, axis=0).astype(np.int16)
    max_sim = S[wins, np.arange(M)].astype(np.float32)

    counts = {name: int((wins == i).sum()) for i, name in enumerate(args.datasets)}
    weights = {name: counts[name] / float(M) for name in args.datasets}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "weights.json").write_text(json.dumps(weights, indent=2))
    (outdir / "counts.json").write_text(json.dumps(counts, indent=2))
    
    # Store the per-query winner (index into args.datasets) and its winning similarity.
    np.save(outdir / "wins.npy", wins)
    np.save(outdir / "max_sim.npy", max_sim)

    print(json.dumps({"M": M, "weights": weights}, indent=2))


if __name__ == "__main__":
    main()