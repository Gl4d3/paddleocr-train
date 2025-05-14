# PaddleOCR Training Scripts

This directory contains scripts for training OCR models using PaddleOCR. These scripts focus on the model training aspects and minimize data manipulation, assuming that data preparation is handled separately.

## Scripts Overview

### 1. `local_training.py`

A streamlined script for training detection and recognition models locally. This script:

- Assumes data is already prepared and formatted correctly
- Focuses solely on model training without modifying datasets
- Trains detection models using the teacher-student approach
- Trains recognition models
- Provides detailed logging
- Allows for flexible configuration via command-line arguments

#### Usage:

```bash
# Train both detection and recognition models
python model_training/scripts/training/local_training.py 

# Train only detection model
python model_training/scripts/training/local_training.py --det_only

# Train only recognition model
python model_training/scripts/training/local_training.py --rec_only

# Configure training parameters
python model_training/scripts/training/local_training.py \
    --gpu_ids=0 \
    --det_dataset_dir=dataset/det_dataset_1 \
    --rec_dataset_dir=dataset/rec_dataset_1 \
    --train_data_dir=train_data/meter_detection \
    --max_det_epochs=50 \
    --max_rec_epochs=50 \
    --det_batch_size=8 \
    --rec_batch_size=64 \
    --det_lr=0.001 \
    --rec_lr=0.001
```

### 2. `validate_dataset.py`

A script that validates dataset readiness for training without modifying any data. This script:

- Checks for existence of required files and directories
- Validates formats of annotation and label files
- Verifies image file existence and integrity (optionally)
- Provides detailed reports on any issues found
- Does NOT modify any data

#### Usage:

```bash
# Validate datasets with default paths
python model_training/scripts/training/validate_dataset.py

# Validate datasets with custom paths
python model_training/scripts/training/validate_dataset.py \
    --det_dataset_dir=path/to/detection/dataset \
    --rec_dataset_dir=path/to/recognition/dataset \
    --train_data_dir=path/to/train/data

# Also check image file integrity (slower)
python model_training/scripts/training/validate_dataset.py --check_images
```

### 3. `kaggle_training.py` and others

These scripts provide alternative approaches for training, including on Kaggle:

- `kaggle_training.py`: Training on Kaggle with MLflow integration
- `kaggle_notebook_training.py`: Template for Kaggle notebook training
- `kaggle_api_trigger.py`: Trigger Kaggle training via API

## Dataset Format Requirements

### Detection Dataset

```
dataset/det_dataset_1/
├── images/
│   └── (image files)
├── train_verified.txt
└── test_verified.txt
```

Each line in the txt files follows this format:
```
image_path\t[{"transcription": "text", "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}]
```

### Recognition Dataset

```
dataset/rec_dataset_1/
├── images/
│   └── (image files)
├── train_list.txt
└── test_list.txt
```

Each line in the txt files follows this format:
```
image_name.jpg\ttext_label
```

## Training Process

The training pipeline uses core PaddleOCR components (`ppocr` and `tools` directories) to:

1. **Detection Model Training**:
   - Train a teacher model (ResNet50 backbone)
   - Train a student model (MobileNetV3 backbone) using knowledge distillation
   - Save models in `model_training/det_train/output/`

2. **Recognition Model Training**:
   - Train a recognition model with pretrained weights
   - Save models in `model_training/rec_train/output/`

## Best Practices

1. **Validate First**: Always validate your datasets before training
2. **Start Small**: Begin with fewer epochs (e.g., 10-20) for testing
3. **Monitor Resources**: Check GPU memory usage with `nvidia-smi`
4. **Check Logs**: Review `training.log` for detailed information
5. **Evaluate Models**: Test models on sample images after training 