#!/bin/bash

# Set up environment
echo "Setting up environment..."
mkdir -p pretrain_models
mkdir -p model_training/det_train/output/meter_teacher
mkdir -p model_training/det_train/output/meter_student
mkdir -p train_data/meter_detection/images

# Convert the dataset to PaddleOCR format
echo "Converting dataset format..."
python3 model_training/det_train/scripts/convert_dataset.py --dataset_dir=dataset/det_dataset_1 --output_dir=train_data/meter_detection

# Create symbolic links for images if they don't exist
if [ ! "$(ls -A train_data/meter_detection/images)" ]; then
  echo "Creating symbolic links for images..."
  ln -sf "$(pwd)/dataset/det_dataset_1/images/"* train_data/meter_detection/images/
fi

# Download pretrained model for teacher
if [ ! -f "pretrain_models/ResNet50_vd_ssld_pretrained.pdparams" ]; then
  echo "Downloading pretrained model for teacher..."
  wget -P pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/ResNet50_vd_ssld_pretrained.pdparams
fi

# Create custom configuration for meter reading dataset
echo "Creating custom configurations..."
cp configs/det/PP-OCRv3/PP-OCRv3_det_dml.yml model_training/det_train/configs/meter_det_teacher.yml
cp configs/det/PP-OCRv3/PP-OCRv3_det_cml.yml model_training/det_train/configs/meter_det_student.yml

# Modify the configuration files to use the meter reading dataset
sed -i 's|data_dir: ./train_data/icdar2015/text_localization/|data_dir: ./train_data/meter_detection/|g' model_training/det_train/configs/meter_det_teacher.yml
sed -i 's|label_file_list:.*|label_file_list:\n      - ./train_data/meter_detection/train_label.txt|g' model_training/det_train/configs/meter_det_teacher.yml
sed -i 's|./train_data/icdar2015/text_localization/test_icdar2015_label.txt|./train_data/meter_detection/test_label.txt|g' model_training/det_train/configs/meter_det_teacher.yml
sed -i 's|save_model_dir: ./output/ch_db_mv3/|save_model_dir: ./model_training/det_train/output/meter_teacher/|g' model_training/det_train/configs/meter_det_teacher.yml

sed -i 's|data_dir: ./train_data/icdar2015/text_localization/|data_dir: ./train_data/meter_detection/|g' model_training/det_train/configs/meter_det_student.yml
sed -i 's|label_file_list:.*|label_file_list:\n      - ./train_data/meter_detection/train_label.txt|g' model_training/det_train/configs/meter_det_student.yml
sed -i 's|./train_data/icdar2015/text_localization/test_icdar2015_label.txt|./train_data/meter_detection/test_label.txt|g' model_training/det_train/configs/meter_det_student.yml
sed -i 's|save_model_dir: ./output/ch_PP-OCR_v3_det/|save_model_dir: ./model_training/det_train/output/meter_student/|g' model_training/det_train/configs/meter_det_student.yml

# Copy configs to main directory for training
cp model_training/det_train/configs/meter_det_teacher.yml configs/det/
cp model_training/det_train/configs/meter_det_student.yml configs/det/

# Train models
echo "=============================="
echo "Step 1: Training teacher model"
echo "=============================="
python3 tools/train.py -c configs/det/meter_det_teacher.yml \
        -o Global.pretrained_model=pretrain_models/ResNet50_vd_ssld_pretrained.pdparams \
           Global.save_model_dir=./model_training/det_train/output/meter_teacher/

echo "=============================="
echo "Step 2: Training student model"
echo "=============================="
python3 tools/train.py -c configs/det/meter_det_student.yml \
        -o Architecture.Models.Teacher.pretrained=./model_training/det_train/output/meter_teacher/best_accuracy \
           Global.save_model_dir=./model_training/det_train/output/meter_student/

echo "Training completed!"
echo "Teacher model saved to: ./model_training/det_train/output/meter_teacher/"
echo "Student model saved to: ./model_training/det_train/output/meter_student/"
echo ""
echo "To evaluate the model, run:"
echo "python3 tools/eval.py -c configs/det/meter_det_student.yml -o Global.checkpoints=./model_training/det_train/output/meter_student/best_accuracy" 