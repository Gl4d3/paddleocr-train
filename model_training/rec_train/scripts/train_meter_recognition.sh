#!/bin/bash

# Exit script on any error
set -e

# Set working directory to PaddleOCR root
cd "$(dirname "$0")/../../.."

# Create required directories
echo "Setting up directories..."
mkdir -p pretrain_models
mkdir -p model_training/rec_train/output/meter_rec
mkdir -p model_training/rec_train/output/meter_rec_distillation
mkdir -p dataset/rec_dataset_1/images
mkdir -p configs/rec/meter

# Check if recognition dataset exists, if not, create it from detection dataset
if [ ! -f "dataset/rec_dataset_1/train_list.txt" ]; then
  echo "Recognition dataset not found. Checking for detection dataset..."
  
  if [ ! -f "train_data/meter_detection/train_label.txt" ]; then
    echo "Error: Detection dataset not found. Please run detection training first or prepare the detection dataset."
    echo "Expected file: train_data/meter_detection/train_label.txt"
    exit 1
  fi
  
  echo "Creating recognition dataset from detection results..."
  python3 model_training/rec_train/scripts/extract_meter_readings.py \
    --det_data_dir=dataset/det_dataset_1 \
    --rec_output_dir=dataset/rec_dataset_1 \
    --train_data=train_data/meter_detection/train_label.txt \
    --test_data=train_data/meter_detection/test_label.txt
fi

# Ensure the dictionary file exists
if [ ! -f "model_training/rec_train/meter_dict.txt" ]; then
  echo "Error: Character dictionary file not found: model_training/rec_train/meter_dict.txt"
  exit 1
fi

# Visualize the recognition dataset
echo "Visualizing recognition dataset..."
python3 model_training/rec_train/scripts/visualize_rec_dataset.py \
  --data_dir=dataset/rec_dataset_1 \
  --output_dir=model_training/rec_train/visualizations

# Download pretrained model if needed
PRETRAINED_MODEL_DIR="pretrain_models/en_PP-OCRv3_rec_train"
if [ ! -d "$PRETRAINED_MODEL_DIR" ]; then
  echo "Downloading pretrained recognition model..."
  if [ ! -f "pretrain_models/en_PP-OCRv3_rec_train.tar" ]; then
    wget -P pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar
  fi
  
  # Extract the model
  cd pretrain_models
  tar -xf en_PP-OCRv3_rec_train.tar
  cd ..
  
  if [ ! -d "$PRETRAINED_MODEL_DIR" ]; then
    echo "Error: Failed to download or extract pretrained model to $PRETRAINED_MODEL_DIR"
    exit 1
  fi
fi

# Copy config files to main configs directory
echo "Copying configuration files..."
cp model_training/rec_train/configs/meter_PP-OCRv3_rec.yml configs/rec/meter/
cp model_training/rec_train/configs/meter_PP-OCRv3_rec_distillation.yml configs/rec/meter/
cp model_training/rec_train/meter_dict.txt ppocr/utils/meter_dict.txt

# Display dataset statistics
TRAIN_COUNT=$(wc -l < dataset/rec_dataset_1/train_list.txt)
TEST_COUNT=$(wc -l < dataset/rec_dataset_1/test_list.txt)
echo "Dataset statistics:"
echo "- Training samples: $TRAIN_COUNT"
echo "- Test samples: $TEST_COUNT"

# Check and notify about labeling
echo "======================================================================="
echo "IMPORTANT: Before training, make sure to update the labels in:"
echo "- dataset/rec_dataset_1/train_list.txt"
echo "- dataset/rec_dataset_1/test_list.txt"
echo ""
echo "The default labels are placeholders (###). These must be replaced"
echo "with actual meter readings for the model to learn correctly."
echo "======================================================================="
read -p "Have you updated the labels with actual meter readings? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Please update the labels before training. Exiting..."
  exit 1
fi

# Train the recognition model
echo "=============================="
echo "Training recognition model"
echo "=============================="

# Choose between regular and distillation training
echo "Select training mode:"
echo "1) Regular recognition training"
echo "2) Distillation-based recognition training"
read -p "Enter your choice (1/2): " -n 1 -r
echo

if [[ $REPLY =~ ^[1]$ ]]; then
  # Regular training
  echo "Starting regular recognition training..."
  python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0' tools/train.py \
    -c configs/rec/meter/meter_PP-OCRv3_rec.yml \
    -o Global.pretrained_model=pretrain_models/en_PP-OCRv3_rec_train/best_accuracy \
    Global.character_dict_path=model_training/rec_train/meter_dict.txt \
    Global.save_model_dir=./model_training/rec_train/output/meter_rec
else
  # Distillation training
  echo "Starting distillation-based recognition training..."
  python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0' tools/train.py \
    -c configs/rec/meter/meter_PP-OCRv3_rec_distillation.yml \
    -o Architecture.Models.Teacher.pretrained=pretrain_models/en_PP-OCRv3_rec_train/best_accuracy \
    Architecture.Models.Student.pretrained=pretrain_models/en_PP-OCRv3_rec_train/best_accuracy \
    Global.character_dict_path=model_training/rec_train/meter_dict.txt \
    Global.save_model_dir=./model_training/rec_train/output/meter_rec_distillation
fi

echo "Training completed!"
echo ""
echo "To evaluate the model, run:"
echo "python3 tools/eval.py -c configs/rec/meter/meter_PP-OCRv3_rec.yml -o Global.checkpoints=./model_training/rec_train/output/meter_rec/best_accuracy"
echo ""
echo "For inference on cropped meter images:"
echo "python3 tools/infer_rec.py -c configs/rec/meter/meter_PP-OCRv3_rec.yml -o Global.infer_img=path/to/cropped/image.jpg Global.checkpoints=./model_training/rec_train/output/meter_rec/best_accuracy" 