from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dw.npy import load_for_faiss


def _require_faiss():
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "FAISS is required for retrieval. Install with: pip install .[faiss]"
        ) from e
    return faiss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_emb", required=True, help="Reference emb_bf16.npy (OpenImages)")
    ap.add_argument("--cand_emb", required=True, help="Candidate emb_bf16.npy")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load with timing
    start = time.time()
    R = load_for_faiss(args.ref_emb).astype(np.float32, copy=False)
    C = load_for_faiss(args.cand_emb).astype(np.float32, copy=False)
    print(f"Loaded {R.shape[0]:,} ref, {C.shape[0]:,} cand in {time.time()-start:.1f}s")

    faiss = _require_faiss()
    
    # Use GPU if available
    index = faiss.IndexFlatIP(C.shape[1])
    index.add(C)
    
    if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
        print("Using GPU...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Search with progress bar
    print(f"Searching {R.shape[0]:,} queries...")
    nn_sim, nn_idx = index.search(R, 1)
    
    np.save(outdir / "nn_sim.npy", nn_sim.reshape(-1))
    np.save(outdir / "nn_idx.npy", nn_idx.reshape(-1))
    print(f"Wrote {outdir} in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()