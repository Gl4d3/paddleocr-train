# Meter Reading Detection with PP-OCRv3

This project demonstrates how to train a PP-OCRv3 detection model for meter reading detection using the PaddleOCR framework.

## Dataset Structure

The dataset folder should be structured as follows:

```
dataset/
└── det_dataset_1/
    ├── train_verified.txt
    ├── test_verified.txt
    └── images/
        └── (image files)
```

Each line in the txt files follows this format:
```
image_path\t[{"transcription": "meter_display", "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}]
```

## Setup and Processing Scripts

The project includes the following scripts in the `scripts/` directory:

1. **test_dataset.py**: Visualizes random samples from the dataset with bounding boxes
   ```
   python3 model_training/det_train/scripts/test_dataset.py
   ```
   This will generate visualizations in the `model_training/det_train/visualizations/` directory.

2. **convert_dataset.py**: Converts the dataset to PaddleOCR format
   ```
   python3 model_training/det_train/scripts/convert_dataset.py --dataset_dir=dataset/det_dataset_1 --output_dir=train_data/meter_detection
   ```
   This creates the following files:
   - `train_data/meter_detection/train_label.txt`
   - `train_data/meter_detection/test_label.txt`

3. **train_meter_detection.sh**: Complete script for training the model
   ```
   ./model_training/det_train/scripts/train_meter_detection.sh
   ```
   This script:
   - Creates necessary directories
   - Converts the dataset
   - Creates symbolic links for images
   - Downloads pretrained models
   - Creates and modifies configuration files
   - Trains teacher and student models

## Training Process

The training process has two steps:

1. **Teacher Model Training**:
   - Uses a ResNet50_vd backbone with LKPAN and DBHead
   - Uses Deep Mutual Learning (DML) distillation method
   - Configuration file: `configs/meter_det_teacher.yml`

2. **Student Model Training**:
   - Uses a lightweight MobileNetV3 backbone with RSEFPN
   - Uses Collaborative Mutual Learning (CML) distillation method
   - Learns from the teacher model
   - Configuration file: `configs/meter_det_student.yml`

To train models individually:

```bash
# Train teacher model
python3 tools/train.py -c configs/det/meter_det_teacher.yml \
        -o Global.pretrained_model=pretrain_models/ResNet50_vd_ssld_pretrained.pdparams \
           Global.save_model_dir=./model_training/det_train/output/meter_teacher/

# Train student model
python3 tools/train.py -c configs/det/meter_det_student.yml \
        -o Architecture.Models.Teacher.pretrained=./model_training/det_train/output/meter_teacher/best_accuracy \
           Global.save_model_dir=./model_training/det_train/output/meter_student/
```

## Evaluation

To evaluate the trained model:

```bash
python3 tools/eval.py -c configs/det/meter_det_student.yml -o Global.checkpoints=./model_training/det_train/output/meter_student/best_accuracy
```

## Inference

To perform inference on a single image:

```bash
python3 tools/infer_det.py -c configs/det/meter_det_student.yml -o Global.infer_img=path/to/your/image.jpg Global.checkpoints=./model_training/det_train/output/meter_student/best_accuracy
```

## Understanding Teacher-Student Models

The teacher-student approach (knowledge distillation) is used to create efficient and accurate models:

1. **Teacher Model**: A large and accurate model (ResNet50) that captures complex patterns but may be too large for deployment
2. **Student Model**: A lightweight model (MobileNetV3) that learns from the teacher and is suitable for deployment
3. **Benefits**:
   - Better accuracy than training the lightweight model directly
   - Smaller model size for efficient deployment
   - Faster inference speed

For more information about PP-OCRv3, refer to the [PaddleOCR documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_en/PP-OCRv3_introduction_en.md). 


## Instructions

To run this  