#!/bin/bash
module load cuda/12.8
cd /home/bunne/projects/dataset_weighting_nn
source .venv/bin/activate

mkdir /globalwork/bunne/data/openimgs
wget https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv

python scripts/make_openimages_manifest.py \
  --csv /globalwork/bunne/data/preprocessed/image_ids_and_rotation.csv \
  --split train \
  --fraction 0.01 \
  --seed 0 \
  --out /globalwork/bunne/data/openimgs/manifest_seed0.txt

python scripts/openimages_downloader.py \
  /globalwork/bunne/data/openimgs/manifest_seed0.txt \
  --download_folder /globalwork/bunne/data/openimgs/images \
  --num_processes 32

dw-extract-embeddings \
  --config configs/default.yaml \
  --dataset openimages \
  --adapter openimages_manifest \
  --manifest /globalwork/bunne/data/openimgs/manifest_seed0.txt \
  --root /globalwork/bunne/data/openimgs/images

dw-extract-embeddings \
  --config configs/default.yaml \
  --dataset hypersim_old \
  --adapter hypersim \
  --root /globalwork/bunne/data/hypersim

dw-extract-embeddings \
  --config configs/default.yaml \
  --dataset gta_sfm \
  --adapter gta_sfm \
  --root /globalwork/bunne/data/GTA-Sfm/data/gta_sfm_clean/train

dw-extract-embeddings \
  --config configs/default.yaml \
  --dataset scannetpp \
  --adapter image_glob \
  --root /globalwork/bunne/data/scannetpp/data \
  --pattern "*/dslr/undistorted_images/*.JPG"

dw-retrieve-1nn \
  --ref_emb artifacts/embeddings/openimages/emb_bf16.npy \
  --cand_emb artifacts/embeddings/hypersim/emb_bf16.npy \
  --outdir artifacts/retrieval/hypersim

dw-retrieve-1nn \
  --ref_emb artifacts/embeddings/openimages/emb_bf16.npy \
  --cand_emb artifacts/embeddings/gta_sfm/emb_bf16.npy \
  --outdir artifacts/retrieval/gta_sfm

dw-retrieve-1nn \
  --ref_emb artifacts/embeddings/openimages/emb_bf16.npy \
  --cand_emb artifacts/embeddings/scannetpp/emb_bf16.npy \
  --outdir artifacts/retrieval/scannetpp

dw-aggregate-weights \
  --datasets hypersim gta_sfm scannetpp \
  --nn_sims \
    artifacts/retrieval/hypersim/nn_sim.npy \
    artifacts/retrieval/gta_sfm/nn_sim.npy \
    artifacts/retrieval/scannetpp/nn_sim.npy \
  --outdir artifacts/weights/test