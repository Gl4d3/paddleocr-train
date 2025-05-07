# Meter Reading Recognition Model Training

This directory contains the scripts and configurations for training a PP-OCRv3 recognition model for meter reading digits and values.

## Overview

After detecting meter display regions with the detection model, the recognition model extracts the actual text/numbers from these regions. The model is based on PP-OCRv3's SVTR_LCNet architecture which combines:

- A lightweight MobileNetV1Enhance backbone
- SVTR (Spatial Vision Transformer) neck
- Multiple prediction heads (CTC and SAR)

## Directory Structure

```
rec_train/
├── configs/                     # Configuration files
│   ├── meter_PP-OCRv3_rec.yml        # Regular recognition config
│   └── meter_PP-OCRv3_rec_distillation.yml  # Distillation-based config
├── meter_dict.txt               # Character dictionary for meter readings
├── output/                      # Training output directory
│   ├── meter_rec/               # Regular training output 
│   └── meter_rec_distillation/  # Distillation training output
├── pretrained_models/           # Pretrained models for initialization
├── scripts/                     # Utility scripts
│   ├── extract_meter_readings.py     # Creates recognition dataset from detection results
│   ├── visualize_rec_dataset.py      # Visualizes the recognition dataset
│   └── train_meter_recognition.sh    # Training script
└── visualizations/              # Recognition results visualization
```

## Steps for Training Recognition Model

### 1. Prepare the Dataset

First, extract cropped meter reading regions from the detection dataset:

```bash
python3 model_training/rec_train/scripts/extract_meter_readings.py \
  --det_data_dir=dataset/det_dataset_1 \
  --rec_output_dir=dataset/rec_dataset_1 \
  --train_data=train_data/meter_detection/train_label.txt \
  --test_data=train_data/meter_detection/test_label.txt
```

This script:
- Extracts text regions based on detection annotations
- Creates cropped images in `dataset/rec_dataset_1/images/`
- Creates label files with placeholders in `dataset/rec_dataset_1/`

### 2. Label the Dataset

**Important**: After extraction, you need to manually add the actual text labels to:
- `dataset/rec_dataset_1/train_list.txt`
- `dataset/rec_dataset_1/test_list.txt`

Replace the placeholder `###` labels with actual meter readings.

The format of these files should be:
```
image_name.jpg\ttext_label
```

For example:
```
meter_01_roi_0.jpg	12.5
meter_02_roi_1.jpg	45
meter_03_roi_0.jpg	6.8
```

### 3. Visualize the Dataset

```bash
python3 model_training/rec_train/scripts/visualize_rec_dataset.py \
  --data_dir=dataset/rec_dataset_1 \
  --output_dir=model_training/rec_train/visualizations
```

This helps verify that your cropped regions and labels are correct.

### 4. Train the Model

The easiest way to train is to use the provided script:

```bash
./model_training/rec_train/scripts/train_meter_recognition.sh
```

This script will:
1. Check if the dataset exists; create it if necessary
2. Visualize the dataset
3. Download and extract the pretrained model
4. Copy configuration files to the correct locations
5. Ask for confirmation that labels have been updated
6. Prompt for training mode (regular or distillation)
7. Start training

### 5. Training Approaches

You can choose between two training approaches:

#### A. Regular Recognition Training

Regular supervised training directly optimizes the model on the labeled data:

```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0' tools/train.py \
  -c configs/rec/meter/meter_PP-OCRv3_rec.yml \
  -o Global.pretrained_model=pretrain_models/en_PP-OCRv3_rec_train/best_accuracy \
  Global.character_dict_path=model_training/rec_train/meter_dict.txt \
  Global.save_model_dir=./model_training/rec_train/output/meter_rec
```

This approach:
- Uses English PP-OCRv3 recognition model as initialization
- Is appropriate for smaller datasets or when you have high-quality labels

#### B. Distillation-based Training

Knowledge distillation creates a student model that learns from both the labeled data and a teacher model:

```bash
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0' tools/train.py \
  -c configs/rec/meter/meter_PP-OCRv3_rec_distillation.yml \
  -o Architecture.Models.Teacher.pretrained=pretrain_models/en_PP-OCRv3_rec_train/best_accuracy \
  Architecture.Models.Student.pretrained=pretrain_models/en_PP-OCRv3_rec_train/best_accuracy \
  Global.character_dict_path=model_training/rec_train/meter_dict.txt \
  Global.save_model_dir=./model_training/rec_train/output/meter_rec_distillation
```

This approach:
- Uses knowledge distillation between teacher and student
- Offers better performance with limited labeled data
- Is more robust to variations in meter displays
- Employs multiple distillation losses (feature-level, logit-level)

## Character Dictionary

The `meter_dict.txt` file contains the allowed characters for meter readings:
- Digits: 0-9
- Special characters: ., -, /, :

If your meter readings contain additional characters, add them to this file.

## Evaluation and Inference

### Evaluation

To evaluate the trained model:

```bash
python3 tools/eval.py -c configs/rec/meter/meter_PP-OCRv3_rec.yml \
  -o Global.checkpoints=./model_training/rec_train/output/meter_rec/best_accuracy \
  Global.character_dict_path=model_training/rec_train/meter_dict.txt
```

### Inference on Cropped Images

For inference on new cropped meter images:

```bash
python3 tools/infer_rec.py -c configs/rec/meter/meter_PP-OCRv3_rec.yml \
  -o Global.infer_img=path/to/cropped/image.jpg \
  Global.checkpoints=./model_training/rec_train/output/meter_rec/best_accuracy \
  Global.character_dict_path=model_training/rec_train/meter_dict.txt
```

## Integration with Detection Model

After training both detection and recognition models, you can use the complete OCR pipeline:

```bash
python3 tools/infer.py \
  -c configs/det/meter/meter_det_student.yml \
  -o Global.infer_img=path/to/image.jpg \
  Global.det_model_dir=./model_training/det_train/output/meter_student \
  Global.rec_model_dir=./model_training/rec_train/output/meter_rec \
  Global.rec_char_dict_path=./model_training/rec_train/meter_dict.txt
```

## Troubleshooting

If you encounter issues during training:

1. **GPU Memory Issues**: Reduce batch size in the config file
2. **Label Issues**: Ensure all labels match the characters in meter_dict.txt
3. **Path Issues**: Make sure all paths are correct relative to the PaddleOCR root directory
4. **Pretrained Model**: Verify the pretrained model was downloaded and extracted correctly 