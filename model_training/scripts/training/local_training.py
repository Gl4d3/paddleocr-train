#!/usr/bin/env python
# coding: utf-8

"""
PaddleOCR Training Pipeline for Local Testing
- Simplified training pipeline for local testing
- Assumes data is already prepared
- Focuses on model training only
"""

import os
import sys
import subprocess
import time
import argparse
import paddle
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger("paddleocr-training")

# Parse arguments for flexible configuration
parser = argparse.ArgumentParser(description='PaddleOCR Training Pipeline for Local Testing')
parser.add_argument('--det_only', action='store_true', help='Run only detection training')
parser.add_argument('--rec_only', action='store_true', help='Run only recognition training')
parser.add_argument('--det_dataset_dir', type=str, default='dataset/det_dataset_1', help='Detection dataset directory')
parser.add_argument('--rec_dataset_dir', type=str, default='dataset/rec_dataset_1', help='Recognition dataset directory')
parser.add_argument('--train_data_dir', type=str, default='train_data/meter_detection', help='Training data directory')
parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs to use')
parser.add_argument('--max_det_epochs', type=int, default=50, help='Max detection training epochs')
parser.add_argument('--max_rec_epochs', type=int, default=50, help='Max recognition training epochs')
parser.add_argument('--det_batch_size', type=int, default=8, help='Detection batch size')
parser.add_argument('--rec_batch_size', type=int, default=64, help='Recognition batch size')
parser.add_argument('--det_lr', type=float, default=0.001, help='Detection learning rate')
parser.add_argument('--rec_lr', type=float, default=0.001, help='Recognition learning rate')
args = parser.parse_args()

def setup_environment():
    """Setup the training environment and ensure correct directory structure."""
    # Get the absolute path of the current script
    current_script = os.path.abspath(__file__)
    
    # Local environment - navigate to PaddleOCR root
    # Go up from model_training/scripts/training to PaddleOCR root
    root_dir = os.path.abspath(os.path.join(os.path.dirname(current_script), "../../.."))
    os.chdir(root_dir)
    
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Verify essential directories exist
    required_dirs = ['ppocr', 'tools']
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    if missing_dirs:
        raise RuntimeError(f"Missing required directories: {', '.join(missing_dirs)}")
    
    # Create output directories
    os.makedirs("model_training/det_train/output/meter_teacher", exist_ok=True)
    os.makedirs("model_training/det_train/output/meter_student", exist_ok=True)
    os.makedirs("model_training/rec_train/output/meter_rec", exist_ok=True)
    
    # Ensure directories exist
    for dir_path in [args.det_dataset_dir, args.rec_dataset_dir, args.train_data_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create links for images if needed
    train_data_images = os.path.join(args.train_data_dir, "images")
    if not os.path.exists(train_data_images):
        os.makedirs(train_data_images, exist_ok=True)

    # Verify data existence
    train_label_path = os.path.join(args.train_data_dir, "train_label.txt")
    test_label_path = os.path.join(args.train_data_dir, "test_label.txt")
    rec_train_path = os.path.join(args.rec_dataset_dir, "train_list.txt")
    rec_test_path = os.path.join(args.rec_dataset_dir, "test_list.txt")
    
    if not os.path.exists(train_label_path) or not os.path.exists(test_label_path):
        logger.warning(f"Detection training data not found at {train_label_path} or {test_label_path}")
        logger.warning("This script assumes data is already prepared. Please prepare data first.")
    
    if not os.path.exists(rec_train_path) or not os.path.exists(rec_test_path):
        logger.warning(f"Recognition training data not found at {rec_train_path} or {rec_test_path}")
        logger.warning("This script assumes data is already prepared. Please prepare data first.")

def get_character_dict():
    """Get or create character dictionary for recognition model"""
    dict_path = "model_training/rec_train/meter_dict.txt"
    
    if not os.path.exists(dict_path):
        logger.info(f"Character dictionary not found: {dict_path}")
        logger.info("Creating a default dictionary with digits and common symbols...")
        
        # Create default dictionary
        default_chars = "0123456789.,-/:"
        os.makedirs(os.path.dirname(dict_path), exist_ok=True)
        
        with open(dict_path, 'w') as f:
            for char in default_chars:
                f.write(f"{char}\n")
        
        logger.info(f"Created default dictionary with {len(default_chars)} characters")
    else:
        # Read existing dictionary
        with open(dict_path, 'r') as f:
            chars = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Character dictionary found with {len(chars)} characters")
    
    return dict_path

def download_det_pretrained_model():
    """Download pretrained detection model if not exists"""
    pretrained_model = "pretrain_models/det_mv3_db_v2.0_train/best_accuracy"
    if not os.path.exists("pretrain_models/det_mv3_db_v2.0_train"):
        logger.info("Downloading pretrained detection model...")
        url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar"
        os.makedirs("pretrain_models", exist_ok=True)
        subprocess.run(f"wget {url} -P pretrain_models/", shell=True)
        subprocess.run(f"tar -xf pretrain_models/det_mv3_db_v2.0_train.tar -C pretrain_models/", shell=True)
    return pretrained_model

def download_rec_pretrained_model():
    """Download pretrained recognition model if not exists"""
    pretrained_model = "pretrain_models/en_PP-OCRv3_rec_train/best_accuracy"
    if not os.path.exists("pretrain_models/en_PP-OCRv3_rec_train"):
        logger.info("Downloading pretrained recognition model...")
        url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar"
        os.makedirs("pretrain_models", exist_ok=True)
        subprocess.run(f"wget {url} -P pretrain_models/", shell=True)
        subprocess.run(f"tar -xf pretrain_models/en_PP-OCRv3_rec_train.tar -C pretrain_models/", shell=True)
    return pretrained_model

def train_detection_model():
    """Train detection model with teacher-student approach"""
    # Download pretrained model
    pretrained_model = download_det_pretrained_model()
    
    # Generate training command for teacher model
    teacher_output_dir = "./model_training/det_train/output/meter_teacher"
    
    logger.info("Starting teacher model training...")
    teacher_cmd = f"""python -m paddle.distributed.launch --gpus='{args.gpu_ids}' tools/train.py \
        -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_teacher.yml \
        -o Global.pretrained_model={pretrained_model} \
        Global.save_model_dir={teacher_output_dir} \
        Train.dataset.data_dir={args.train_data_dir} \
        Train.dataset.label_file_list=['{os.path.join(args.train_data_dir, "train_label.txt")}'] \
        Eval.dataset.data_dir={args.train_data_dir} \
        Eval.dataset.label_file_list=['{os.path.join(args.train_data_dir, "test_label.txt")}'] \
        Train.loader.batch_size_per_card={args.det_batch_size} \
        Optimizer.lr.values=[{args.det_lr},{args.det_lr/10}] \
        Global.epoch_num={args.max_det_epochs} \
        Global.save_epoch_step=5 \
        Global.eval_batch_step=[0, 500]"""
    
    logger.info(f"Teacher model command: {teacher_cmd}")
    start_time = time.time()
    process = subprocess.Popen(teacher_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Track training output
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.wait()
    training_time = time.time() - start_time
    logger.info(f"Teacher model training completed in {training_time:.2f} seconds")
    
    # Check if teacher model was successfully created
    if not os.path.exists(f"{teacher_output_dir}/best_accuracy.pdparams"):
        logger.error("Teacher model training failed!")
        return False
    
    # Train student model (knowledge distillation)
    student_output_dir = "./model_training/det_train/output/meter_student"
    
    logger.info("Starting student model training...")
    student_cmd = f"""python -m paddle.distributed.launch --gpus='{args.gpu_ids}' tools/train.py \
        -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml \
        -o Architecture.Models.Teacher.pretrained={teacher_output_dir}/best_accuracy \
        Global.save_model_dir={student_output_dir} \
        Train.dataset.data_dir={args.train_data_dir} \
        Train.dataset.label_file_list=['{os.path.join(args.train_data_dir, "train_label.txt")}'] \
        Eval.dataset.data_dir={args.train_data_dir} \
        Eval.dataset.label_file_list=['{os.path.join(args.train_data_dir, "test_label.txt")}'] \
        Train.loader.batch_size_per_card={args.det_batch_size} \
        Optimizer.lr.values=[{args.det_lr},{args.det_lr/10}] \
        Global.epoch_num={args.max_det_epochs} \
        Global.save_epoch_step=5 \
        Global.eval_batch_step=[0, 500]"""
    
    logger.info(f"Student model command: {student_cmd}")
    start_time = time.time()
    process = subprocess.Popen(student_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Track training output
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.wait()
    training_time = time.time() - start_time
    logger.info(f"Student model training completed in {training_time:.2f} seconds")
    
    # Check if student model was successfully created
    if os.path.exists(f"{student_output_dir}/best_accuracy.pdparams"):
        logger.info("Detection model training completed successfully!")
        return True
    else:
        logger.error("Student model training failed!")
        return False

def train_recognition_model():
    """Train recognition model"""
    # Get character dictionary
    dict_path = get_character_dict()
    
    # Download pretrained model
    pretrained_model = download_rec_pretrained_model()
    
    # Generate training command
    output_dir = "./model_training/rec_train/output/meter_rec"
    
    # Check if config exists, if not create it
    config_dir = "configs/rec/meter"
    os.makedirs(config_dir, exist_ok=True)
    if not os.path.exists(f"{config_dir}/meter_PP-OCRv3_rec.yml"):
        logger.info("Creating recognition config file...")
        subprocess.run(f"cp configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml {config_dir}/meter_PP-OCRv3_rec.yml", shell=True)
        
        # Modify config for meter recognition
        with open(f"{config_dir}/meter_PP-OCRv3_rec.yml", 'r') as f:
            config = f.read()
        
        # Update data paths and character dict path
        config = config.replace("Train.dataset.data_dir", f"# Train.dataset.data_dir")
        config = config.replace("Eval.dataset.data_dir", f"# Eval.dataset.data_dir")
        
        with open(f"{config_dir}/meter_PP-OCRv3_rec.yml", 'w') as f:
            f.write(config)
    
    logger.info("Starting recognition model training...")
    cmd = f"""python -m paddle.distributed.launch --gpus='{args.gpu_ids}' tools/train.py \
        -c configs/rec/meter/meter_PP-OCRv3_rec.yml \
        -o Global.pretrained_model={pretrained_model} \
        Global.character_dict_path={dict_path} \
        Global.save_model_dir={output_dir} \
        Train.dataset.data_dir={args.rec_dataset_dir} \
        Train.dataset.label_file_list=['{os.path.join(args.rec_dataset_dir, "train_list.txt")}'] \
        Eval.dataset.data_dir={args.rec_dataset_dir} \
        Eval.dataset.label_file_list=['{os.path.join(args.rec_dataset_dir, "test_list.txt")}'] \
        Train.loader.batch_size_per_card={args.rec_batch_size} \
        Optimizer.lr.values=[{args.rec_lr},{args.rec_lr/10}] \
        Global.epoch_num={args.max_rec_epochs} \
        Global.save_epoch_step=5 \
        Global.eval_batch_step=[0, 500]"""
    
    logger.info(f"Recognition model command: {cmd}")
    start_time = time.time()
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Track training output
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.wait()
    training_time = time.time() - start_time
    logger.info(f"Recognition model training completed in {training_time:.2f} seconds")
    
    # Check if recognition model was successfully created
    if os.path.exists(f"{output_dir}/best_accuracy.pdparams"):
        logger.info("Recognition model training completed successfully!")
        return True
    else:
        logger.error("Recognition model training failed!")
        return False

def main():
    """Main function to run the training pipeline"""
    logger.info("=" * 50)
    logger.info("PaddleOCR Training Pipeline for Local Testing")
    logger.info("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check paddle version and GPU availability
    logger.info(f"Paddle version: {paddle.__version__}")
    logger.info(f"GPU available: {paddle.device.is_compiled_with_cuda()}")
    
    train_success = True
    
    # Run detection training if requested
    if not args.rec_only:
        logger.info("\n" + "=" * 50)
        logger.info("DETECTION MODEL TRAINING")
        logger.info("=" * 50)
        det_success = train_detection_model()
        train_success = train_success and det_success
    
    # Run recognition training if requested
    if not args.det_only:
        logger.info("\n" + "=" * 50)
        logger.info("RECOGNITION MODEL TRAINING")
        logger.info("=" * 50)
        rec_success = train_recognition_model()
        train_success = train_success and rec_success
    
    # Print final status
    if train_success:
        logger.info("\n" + "=" * 50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 50)
    else:
        logger.info("\n" + "=" * 50)
        logger.info("TRAINING COMPLETED WITH ERRORS")
        logger.info("=" * 50)

if __name__ == "__main__":
    main() 