#!/bin/bash
# ============================================================================
# Dataset Weighting Pipeline - Complete Workflow
# 
# This script runs the full pipeline:
# 1. Downloads and prepares OpenImages dataset
# 2. Extracts DINOv2 embeddings for all datasets
# 3. Performs 1-NN retrieval between OpenImages and target datasets
# 4. Aggregates weights for training
#
# ============================================================================

# ============================================================================
# CONFIGURATION - Modify these variables for your environment
# ============================================================================

# Project paths
PROJECT_ROOT="/home/bunne/projects/dataset_weighting_nn"
VENV_PATH="$PROJECT_ROOT/.venv/bin/activate"

# Data storage directory
DATA_ROOT="/globalwork/bunne/data"

# OpenImages configuration
OPENIMAGES_DIR="$DATA_ROOT/openimgs"
OPENIMAGES_CSV_URL="https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv"
OPENIMAGES_SPLIT="train"
OPENIMAGES_FRACTION="0.07"  # ~600k images
OPENIMAGES_TARGET="100000"  # Number to actually download
OPENIMAGES_WORKERS="64"

# Target datasets (adjust paths as needed)
HYPERSIM_DIR="$DATA_ROOT/hypersim"
GTASFM_DIR="$DATA_ROOT/GTA-Sfm/data/gta_sfm_clean/train"
SCANNETPP_DIR="$DATA_ROOT/scannetpp/data"
SCANNETPP_PATTERN="*/dslr/undistorted_images/*.JPG"

# Artifacts output directory
ARTIFACTS_DIR="$PROJECT_ROOT/artifacts"

# ============================================================================
# INITIALIZATION
# ============================================================================

echo "======================================================================="
echo "Starting Dataset Weighting Pipeline"
echo "Date: $(date)"
echo "======================================================================="

# Load CUDA module (adjust for your cluster)
module load cuda/12.8 2>/dev/null || echo "CUDA module not available, continuing..."

# Change to project directory
cd "$PROJECT_ROOT" || { echo "Error: Cannot cd to $PROJECT_ROOT"; exit 1; }

# Activate virtual environment
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "Virtual environment activated"
else
    echo "Warning: Virtual environment not found at $VENV_PATH"
    echo "Continuing with system Python..."
fi

# Create necessary directories
mkdir -p "$OPENIMAGES_DIR"
mkdir -p "$ARTIFACTS_DIR"

# ============================================================================
# STEP 1: Prepare OpenImages Dataset
# ============================================================================

echo ""
echo "======================================================================="
echo "STEP 1: Preparing OpenImages dataset"
echo "======================================================================="

# Download OpenImages CSV if not present
CSV_FILE="$OPENIMAGES_DIR/image_ids_and_rotation.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "Downloading OpenImages CSV..."
    wget -q "$OPENIMAGES_CSV_URL" -O "$CSV_FILE"
    echo "Downloaded CSV to $CSV_FILE"
else
    echo "CSV already exists: $CSV_FILE"
fi

# Create manifest file
MANIFEST_FILE="$OPENIMAGES_DIR/manifest600k.txt"
echo "Creating manifest file..."
python scripts/make_openimages_manifest.py \
    --csv "$CSV_FILE" \
    --split "$OPENIMAGES_SPLIT" \
    --fraction "$OPENIMAGES_FRACTION" \
    --out "$MANIFEST_FILE"

# Download images
IMAGE_DIR="$OPENIMAGES_DIR/images"
MANIFEST_OUT="$OPENIMAGES_DIR/success_manifest.txt"
echo "Downloading OpenImages (target: $OPENIMAGES_TARGET images)..."
python scripts/openimages_downloader_robust.py \
    "$MANIFEST_FILE" \
    --download_folder "$IMAGE_DIR" \
    --target "$OPENIMAGES_TARGET" \
    --num_workers "$OPENIMAGES_WORKERS" \
    --count_existing \
    --success_manifest_out "$MANIFEST_OUT"

# Fix manifest format if needed (replace underscores with slashes)
if [ -f "$MANIFEST_OUT" ]; then
    echo "Fixing manifest format..."
    sed -i 's|_|/|g' "$MANIFEST_OUT"
fi

# ============================================================================
# STEP 2: Extract DINOv2 Embeddings for All Datasets
# ============================================================================

echo ""
echo "======================================================================="
echo "STEP 2: Extracting DINOv2 embeddings"
echo "======================================================================="

# OpenImages embeddings
echo "Extracting OpenImages embeddings..."
dw-extract-embeddings \
    --config configs/default.yaml \
    --dataset openimages \
    --adapter openimages_manifest \
    --manifest "$MANIFEST_OUT" \
    --root "$IMAGE_DIR"

# Hypersim embeddings
echo "Extracting Hypersim embeddings..."
dw-extract-embeddings \
    --config configs/default.yaml \
    --dataset hypersim_old \
    --adapter hypersim \
    --root "$HYPERSIM_DIR"

# GTA-SfM embeddings
echo "Extracting GTA-SfM embeddings..."
dw-extract-embeddings \
    --config configs/default.yaml \
    --dataset gta_sfm \
    --adapter gta_sfm \
    --root "$GTASFM_DIR"

# ScanNet++ embeddings
echo "Extracting ScanNet++ embeddings..."
dw-extract-embeddings \
    --config configs/default.yaml \
    --dataset scannetpp \
    --adapter image_glob \
    --root "$SCANNETPP_DIR" \
    --pattern "$SCANNETPP_PATTERN"

# ============================================================================
# STEP 3: 1-NN Retrieval Between OpenImages and Target Datasets
# ============================================================================

echo ""
echo "======================================================================="
echo "STEP 3: Performing 1-NN retrieval"
echo "======================================================================="

REF_EMB="$ARTIFACTS_DIR/embeddings/openimages/emb_bf16.npy"

# Hypersim retrieval
echo "Retrieving for Hypersim..."
dw-retrieve-1nn \
    --ref_emb "$REF_EMB" \
    --cand_emb "$ARTIFACTS_DIR/embeddings/hypersim/emb_bf16.npy" \
    --outdir "$ARTIFACTS_DIR/retrieval/hypersim"

# GTA-SfM retrieval
echo "Retrieving for GTA-SfM..."
dw-retrieve-1nn \
    --ref_emb "$REF_EMB" \
    --cand_emb "$ARTIFACTS_DIR/embeddings/gta_sfm/emb_bf16.npy" \
    --outdir "$ARTIFACTS_DIR/retrieval/gta_sfm"

# ScanNet++ retrieval
echo "Retrieving for ScanNet++..."
dw-retrieve-1nn \
    --ref_emb "$REF_EMB" \
    --cand_emb "$ARTIFACTS_DIR/embeddings/scannetpp/emb_bf16.npy" \
    --outdir "$ARTIFACTS_DIR/retrieval/scannetpp"

# ============================================================================
# STEP 4: Aggregate Weights for Training
# ============================================================================

echo ""
echo "======================================================================="
echo "STEP 4: Aggregating weights"
echo "======================================================================="

dw-aggregate-weights \
    --datasets hypersim gta_sfm scannetpp \
    --nn_sims \
        "$ARTIFACTS_DIR/retrieval/hypersim/nn_sim.npy" \
        "$ARTIFACTS_DIR/retrieval/gta_sfm/nn_sim.npy" \
        "$ARTIFACTS_DIR/retrieval/scannetpp/nn_sim.npy" \
    --outdir "$ARTIFACTS_DIR/weights/test"

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "======================================================================="
echo "Pipeline completed successfully!"
echo "Finished at: $(date)"
echo "======================================================================="
echo ""
echo "Output summary:"
echo "- OpenImages data: $IMAGE_DIR"
echo "- Embeddings: $ARTIFACTS_DIR/embeddings/"
echo "- Retrieval results: $ARTIFACTS_DIR/retrieval/"
echo "- Final weights: $ARTIFACTS_DIR/weights/test/"
echo "======================================================================="