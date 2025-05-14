#!/bin/bash

# PaddleOCR Training Pipeline Runner
# This script runs the complete training pipeline:
# 1. Validates datasets
# 2. Trains detection and recognition models

set -e  # Exit on any error

# Default parameters
DET_DATASET_DIR="dataset/det_dataset_1"
REC_DATASET_DIR="dataset/rec_dataset_1"
TRAIN_DATA_DIR="train_data/meter_detection"
GPU_IDS="0"
MAX_DET_EPOCHS=50
MAX_REC_EPOCHS=50
DET_BATCH_SIZE=8
REC_BATCH_SIZE=64
DET_LR=0.001
REC_LR=0.001
MODE="full"  # Options: full, det_only, rec_only

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --det_dataset_dir=*)
      DET_DATASET_DIR="${1#*=}"
      shift
      ;;
    --rec_dataset_dir=*)
      REC_DATASET_DIR="${1#*=}"
      shift
      ;;
    --train_data_dir=*)
      TRAIN_DATA_DIR="${1#*=}"
      shift
      ;;
    --gpu_ids=*)
      GPU_IDS="${1#*=}"
      shift
      ;;
    --max_det_epochs=*)
      MAX_DET_EPOCHS="${1#*=}"
      shift
      ;;
    --max_rec_epochs=*)
      MAX_REC_EPOCHS="${1#*=}"
      shift
      ;;
    --det_batch_size=*)
      DET_BATCH_SIZE="${1#*=}"
      shift
      ;;
    --rec_batch_size=*)
      REC_BATCH_SIZE="${1#*=}"
      shift
      ;;
    --det_lr=*)
      DET_LR="${1#*=}"
      shift
      ;;
    --rec_lr=*)
      REC_LR="${1#*=}"
      shift
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --det_dataset_dir=DIR    Detection dataset directory (default: $DET_DATASET_DIR)"
      echo "  --rec_dataset_dir=DIR    Recognition dataset directory (default: $REC_DATASET_DIR)"
      echo "  --train_data_dir=DIR     Training data directory (default: $TRAIN_DATA_DIR)"
      echo "  --gpu_ids=IDS            GPU IDs to use (default: $GPU_IDS)"
      echo "  --max_det_epochs=NUM     Max detection epochs (default: $MAX_DET_EPOCHS)"
      echo "  --max_rec_epochs=NUM     Max recognition epochs (default: $MAX_REC_EPOCHS)"
      echo "  --det_batch_size=NUM     Detection batch size (default: $DET_BATCH_SIZE)"
      echo "  --rec_batch_size=NUM     Recognition batch size (default: $REC_BATCH_SIZE)"
      echo "  --det_lr=NUM             Detection learning rate (default: $DET_LR)"
      echo "  --rec_lr=NUM             Recognition learning rate (default: $REC_LR)"
      echo "  --mode=MODE              Training mode: full, det_only, rec_only (default: $MODE)"
      echo "  --help                   Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the project root directory
cd "$SCRIPT_DIR/../../.."
PROJECT_ROOT=$(pwd)

echo "======================================================================"
echo "PaddleOCR Training Pipeline"
echo "======================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Detection dataset: $DET_DATASET_DIR"
echo "Recognition dataset: $REC_DATASET_DIR"
echo "Training data: $TRAIN_DATA_DIR"
echo "Training mode: $MODE"
echo "GPU IDs: $GPU_IDS"
echo "----------------------------------------------------------------------"

# Step 1: Validate datasets
echo "Step 1: Validating datasets..."
validation_cmd="python3 model_training/scripts/training/validate_dataset.py --det_dataset_dir=$DET_DATASET_DIR --rec_dataset_dir=$REC_DATASET_DIR --train_data_dir=$TRAIN_DATA_DIR"
echo "Running: $validation_cmd"
eval $validation_cmd

# Check if validation was successful
if [ $? -ne 0 ]; then
    echo "❌ Dataset validation failed. Please fix the issues before proceeding."
    exit 1
fi

# Step 2: Train the models
echo ""
echo "Step 2: Training models..."

train_cmd="python3 model_training/scripts/training/local_training.py --det_dataset_dir=$DET_DATASET_DIR --rec_dataset_dir=$REC_DATASET_DIR --train_data_dir=$TRAIN_DATA_DIR --gpu_ids=$GPU_IDS --max_det_epochs=$MAX_DET_EPOCHS --max_rec_epochs=$MAX_REC_EPOCHS --det_batch_size=$DET_BATCH_SIZE --rec_batch_size=$REC_BATCH_SIZE --det_lr=$DET_LR --rec_lr=$REC_LR"

# Add mode-specific flags
if [ "$MODE" = "det_only" ]; then
    train_cmd="$train_cmd --det_only"
elif [ "$MODE" = "rec_only" ]; then
    train_cmd="$train_cmd --rec_only"
fi

echo "Running: $train_cmd"
eval $train_cmd

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "❌ Training failed. Please check the logs for details."
    exit 1
fi

echo ""
echo "======================================================================"
echo "🎉 Training pipeline completed successfully!"
echo "======================================================================"
echo "Models saved to:"
if [ "$MODE" != "rec_only" ]; then
    echo "- Detection teacher model: model_training/det_train/output/meter_teacher/"
    echo "- Detection student model: model_training/det_train/output/meter_student/"
fi
if [ "$MODE" != "det_only" ]; then
    echo "- Recognition model:      model_training/rec_train/output/meter_rec/"
fi
echo ""
echo "Next steps:"
echo "1. Evaluate the models on test data"
echo "2. Export models for inference"
echo "3. Test the models on your own images"
echo "======================================================================" 