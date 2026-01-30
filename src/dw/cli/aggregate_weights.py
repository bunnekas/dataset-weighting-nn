from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True, help="dataset names in same order as nn_sims")
    ap.add_argument("--nn_sims", nargs="+", required=True, help="paths to nn_sim.npy per dataset")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    K = len(args.datasets)
    if len(args.nn_sims) != K:
        raise ValueError("datasets and nn_sims must have same length")

    sims = [np.load(p).astype(np.float32) for p in args.nn_sims]  # each (M,)
    M = sims[0].shape[0]
    for s in sims:
        assert s.shape == (M,)

    S = np.stack(sims, axis=0)  # (K,M)
    wins = np.argmax(S, axis=0).astype(np.int16)  # (M,)
    max_sim = S[wins, np.arange(M)].astype(np.float32)

    counts = {name: int((wins == i).sum()) for i, name in enumerate(args.datasets)}
    weights = {name: counts[name] / float(M) for name in args.datasets}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "weights.json").write_text(json.dumps(weights, indent=2))
    (outdir / "counts.json").write_text(json.dumps(counts, indent=2))
    np.save(outdir / "wins.npy", wins)
    np.save(outdir / "max_sim.npy", max_sim)

    print(json.dumps({"M": M, "weights": weights}, indent=2))

if __name__ == "__main__":
    main()