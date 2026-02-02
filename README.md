# dataset_weighting_nn

Compute dataset mixing weights so that a training mixture best matches the distribution of natural images, following *MoGe Appendix B.1*.

We estimate weights via 1-NN retrieval in DINOv2 ViT-g/14 CLS embedding space, using OpenImages v7 train subsample as the reference distribution.

## Protocol (invariants)

**Embeddings**
- Backbone: `facebookresearch/dinov2`, `dinov2_vitg14`
- Token: CLS
- Similarity: cosine via L2-normalized embeddings + FAISS IndexFlatIP
- Storage: embeddings saved as bf16 (`emb_bf16.npy`), cast to float32 for FAISS

**Image preprocessing**
- Resize: max edge = 512 (no crop)
- Pad: batches so H,W divisible by 14
- Normalize: ImageNet mean/std

## Pipeline

1) **Extract embeddings** for reference + each candidate dataset  
2) **Retrieve 1-NN** from each candidate index for every reference embedding  
3) **Aggregate weights**: each reference sample “votes” for the candidate with the highest similarity

Weights = win frequency over all reference queries.

## Environment / Requirements

This project is tested and intended to run on **Python 3.10.x**.

**Python ≥ 3.12 is not supported** at the moment due to incompatibilities in
the scientific stack (FAISS, torch / torchvision, and some dataset tooling).

**Recommended version**: Python 3.10.14

## Install

```bash
uv venv --python 3.10.14
source .venv/bin/activate
uv pip install -U pip
uv pip install -e .
uv pip install -e /path/to/wai-clone
```

FAISS is required for retrieval. Install one of:
```bash
# CPU version (works everywhere)
uv pip install faiss-cpu

# OR GPU version (Linux only, requires NVIDIA GPU)
uv pip install faiss-gpu
```

## Run (end-to-end)

### 1) Download `image_ids_and_rotation.csv`

This file contains the canonical OpenImages image IDs and is required to
construct any subset manifest.

```bash
mkdir path/to/openimgs
cd path/to/openimgs
wget https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv
```

### 2) Create the subset manifest (image IDs)
We use OpenImages v7 train (1% subsample) as the reference set.

```bash
python scripts/make_openimages_manifest.py \
  --csv path/to/image_ids_and_rotation.csv \
  --split train \
  --fraction 0.07 \
  --seed 0 \
  --out path/to/openimgs/manifest600k.txt
```
Output: `manifest600k.txt`

### 3) Download images referenced by the manifest
```bash
python scripts/openimages_downloader.py \
  path/to/openimgs/manifest600k.txt \
  --download_folder path/to/openimgs/images \
  --target 10000 \
  --num_workers 32 \
  --count_existing \
  --success_manifest_out path/to/openimgs/manifest.txt
```
**Notes:**
- HTTP 404s are expected; missing images are skipped.
- The `--target` flag controls how many images to download.
- Downstream embedding and retrieval only operate on successfully downloaded images.
- The success manifest (manifest.txt) contains the actual downloaded images.

### 4) Extract OpenImages embeddings (reference)

```bash
dw-extract-embeddings \
  --config configs/default.yaml \
  --dataset openimages \
  --adapter openimages_manifest \
  --manifest path/to/openimgs/manifest.txt \
  --root path/to/openimgs/images
```

### 5) Extract candidate embeddings

**Hypersim**
```bash
dw-extract-embeddings \
  --config configs/default.yaml \
  --dataset hypersim \
  --adapter hypersim \
  --root /path/to/hypersim
```

**GTA-SfM**
```bash
dw-extract-embeddings \
  --config configs/default.yaml \
  --dataset gta_sfm \
  --adapter gta_sfm \
  --root /path/to/gta_sfm
```

**ScanNet++ (or any globbed image dataset)**
```bash
dw-extract-embeddings \
  --config configs/default.yaml \
  --dataset scannetpp \
  --adapter image_glob \
  --root /path/to/scannetpp \
  --pattern "*/dslr/undistorted_images/*.JPG"
```

### 6) 1-NN retrieval (OpenImages queries, candidate index)

```bash
dw-retrieve-1nn \
  --ref_emb artifacts/embeddings/openimgs/emb_bf16.npy \
  --cand_emb artifacts/embeddings/hypersim/emb_bf16.npy \
  --outdir artifacts/retrieval/hypersim
```

Repeat for each candidate dataset.

### 7) Aggregate weights

```bash
dw-aggregate-weights \
  --datasets hypersim gta_sfm scannetpp \
  --nn_sims \
    artifacts/retrieval/openimgs/nn_sim.npy \
    artifacts/retrieval/openimgs/nn_sim.npy \
    artifacts/retrieval/openimgs/nn_sim.npy \
  --outdir artifacts/weights
```

Note: For a complete automated pipeline, see run_all.sh which orchestrates all steps.

## Artifacts layout (gitignored)

- `artifacts/embeddings/<dataset>/`
  - `emb_bf16.npy`
  - `paths.jsonl`
  - `meta.json`
- `artifacts/retrieval/<ref>__<cand>/`
  - `nn_sim.npy`
  - `nn_idx.npy`
- `artifacts/weights/<ref>/`
  - `weights.json`
  - `counts.json`
  - `wins.npy`
  - `max_sim.npy`