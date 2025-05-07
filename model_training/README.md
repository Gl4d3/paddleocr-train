# Model Training for Meter Reading OCR

This directory contains the organized training files and configurations for training OCR models on meter reading data using PaddleOCR.

## Directory Structure

```
model_training/
├── det_train/             # Text detection model training
│   ├── configs/           # Configuration files for detection models
│   ├── output/            # Training output directory
│   │   ├── meter_teacher/ # Teacher model checkpoints and logs
│   │   └── meter_student/ # Student model checkpoints and logs
│   ├── pretrained_models/ # Pretrained models for initialization
│   ├── scripts/           # Utility scripts for data preparation and training
│   └── visualizations/    # Dataset visualization outputs
└── rec_train/             # Text recognition model training
    ├── configs/           # Configuration files for recognition models
    ├── meter_dict.txt     # Character dictionary for meter readings
    ├── output/            # Training output directory
    ├── pretrained_models/ # Pretrained models for initialization
    ├── scripts/           # Utility scripts for dataset creation and training
    └── visualizations/    # Dataset visualization outputs
```

## Training Process

The training process is orchestrated through:
1. Detection model training: `model_training/det_train/scripts/train_meter_detection.sh`
2. Recognition model training: `model_training/rec_train/scripts/train_meter_recognition.sh`

These scripts guide you through:

1. Dataset preparation and visualization
2. Format conversion for PaddleOCR
3. Teacher model training (ResNet50 backbone)
4. Student model training (MobileNetV3 backbone)
5. Model evaluation and inference
6. Model export for deployment

## Detection Model (det_train)

The detection model training uses a knowledge distillation approach with two steps:

1. **Teacher Model**: Uses a ResNet50 backbone with LKPAN (Large Kernel PAN) neck and DBHead, trained with Deep Mutual Learning (DML)
2. **Student Model**: Uses a lightweight MobileNetV3 backbone with RSEFPN (Residual SE-FPN) neck and DBHead, trained with Collaborative Mutual Learning (CML)

This approach results in a deployment-friendly model that maintains high accuracy for meter reading detection.

## Recognition Model (rec_train)

The recognition model training also uses knowledge distillation:

1. **Teacher Model**: Based on PP-OCRv3 with SVTR_LCNet architecture
2. **Student Model**: Similar architecture but optimized for deployment

The recognition model is crucial for extracting the actual meter readings from the detected regions. It employs:
- Multiple prediction heads (CTC and SAR) 
- MobileNetV1Enhance backbone
- SVTR (Spatial Vision Transformer) neck

Both regular training and distillation-based approaches are supported.

## Original Data

The original dataset is located in `dataset/dataset_1/` with the following structure:
- `images/`: Contains meter reading images
- `train_verified.txt`: Training set annotations
- `test_verified.txt`: Test set annotations

## Converted Data

The converted dataset for PaddleOCR training is located in:

1. Detection dataset: `train_data/meter_reading/`
   - `images/`: Symbolic links to the original images
   - `train_label.txt`: Converted training annotations in PaddleOCR format
   - `test_label.txt`: Converted test annotations in PaddleOCR format

2. Recognition dataset: `dataset/rec_dataset_1/`
   - `images/`: Cropped text regions from detection results
   - `train_list.txt`: Training annotations for recognition
   - `test_list.txt`: Test annotations for recognition 