# dataset_weighting_nn

Compute dataset mixing weights so that a training mixture best matches the distribution of natural images, following *MoGe Appendix B.1*.

## Pipeline

We estimate weights via 1-NN retrieval in DINOv2 ViT-g/14 CLS embedding space, using OpenImages v7 train subsample as the reference distribution.

<details>
<summary><b>0) Download reference dataset</b></summary>
Download a representative subset as the reference distribution against which candidate datasets will be measured. We use OpenImages v7 train (1% subsample).

### Download `image_ids_and_rotation.csv`

This file contains the canonical OpenImages image IDs and is required to randomly sample a arbitrary subset.

```bash
mkdir path/to/openimgs
cd path/to/openimgs
wget https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv
```

### OpenImages download script

```bash
python scripts/download_openimages.py \
  --csv /path/to/image_ids_and_rotation.csv \
  --out /path/to/openimgs \
  --target 100000 \
  --split train \
  --workers 32 \
  --retries 0
```

### Output structure

```bash
/path/to/openimgs/
├── images/
│   └── *.jpg (downloaded images, named as image_id.jpg)
├── download_log.jsonl
├── success_manifest.txt
└── download_summary.json
```

</details>

<details>
<summary><b>1) Extract embeddings</b></summary>
Compute DINOv2 ViT-g/14 CLS embeddings for both reference and candidate datasets after resizing to 672px max edge and padding to ViT patch alignment.

### Extract OpenImages embeddings (reference set)

```bash
dw-extract-embeddings \
  --config configs/default.yaml \
  --dataset openimages \
  --root path/to/openimgs/images \
  --pattern "*.jpg"
```

### Extract candidate embeddings (any dataset)

For each dataset, specify the `--pattern` argument using glob syntax where `*` matches any single file or directory name, and `**` recursively matches any number of subdirectories.

```bash
# Images in scene-specific folders
--pattern "**/rgb/*.png"
--pattern "**/cam-*/rgb/*.jpg"

# All JPEGs anywhere in dataset
--pattern "**/*.jpg"

# Specific naming convention  
--pattern "image_*.jpeg"
```

```bash
dw-extract-embeddings \
  --config configs/default.yaml \
  --dataset <dataset> \
  --root /path/to/dataset \
  --pattern "**/*.jpg"
```

### Example: ScanNet++

Optionally set `--max-frames-per-scene`, default is all frames.

```bash
dw-extract-embeddings \
  --config configs/default.yaml \
  --dataset scannetpp \
  --root /path/to/scannetpp \
  --pattern "**/dslr/undistorted_images/*.JPG" \
  --max-frames-per-scene 100
```

</details>

<details>
<summary><b>2) Retrieve 1-NN</b></summary>
For each reference embedding, find its nearest neighbor in each candidate dataset using cosine similarity in the normalized embedding space.

### 1-NN retrieval (OpenImages queries, candidate index)

```bash
dw-retrieve-1nn \
  --ref_emb artifacts/embeddings/<ref>/emb.npy \
  --cand_emb artifacts/embeddings/<dataset>/emb.npy \
  --outdir artifacts/retrieval/<ref_dataset>
```

Repeat for each candidate dataset.
</details>

<details>
<summary><b>3) Aggregate weights</b></summary>

Count how often each candidate dataset provides the nearest neighbor; normalize these counts to produce final mixing weights proportional to distribution match.

```bash
dw-aggregate-weights \
  --datasets dataset1 ... datasetN \
  --outdir artifacts/weights
```

Weights = win frequency over all reference queries.
</details>

## Output directory structure

Output directory can be set in `configs/default.yaml`, by default the artifacts are saved at the project root.
```bash
artifacts/
├── embeddings/
│   └── <dataset>/
│       ├── emb.npy
│       ├── paths.jsonl
│       └── meta.json
├── retrieval/
│   └── <ref>_<cand>/
│       ├── nn_sim.npy
│       └── nn_idx.npy
└── weights/
    └── <ref>/
        ├── weights.json
        ├── counts.json
        ├── wins.npy
        └── max_sim.npy
```
