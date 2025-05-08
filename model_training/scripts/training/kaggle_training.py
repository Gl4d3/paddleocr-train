#!/usr/bin/env python
# coding: utf-8

"""
PaddleOCR Training Pipeline for Kaggle
- Combines detection and recognition training
- Integrates MLflow tracking
- Configurable via environment variables
"""

import os
import sys
import glob
import subprocess
import time
import json
import argparse
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
from IPython.display import display, Image
import numpy as np
import paddle

# Parse arguments for flexible configuration
parser = argparse.ArgumentParser(description='PaddleOCR Training Pipeline')
parser.add_argument('--det_only', action='store_true', help='Run only detection training')
parser.add_argument('--rec_only', action='store_true', help='Run only recognition training')
parser.add_argument('--exp_name', type=str, default='paddleocr_training', help='Experiment name')
parser.add_argument('--tracking_uri', type=str, default=None, help='MLflow tracking URI')
parser.add_argument('--det_dataset_dir', type=str, default='dataset/det_dataset_1', help='Detection dataset directory')
parser.add_argument('--rec_dataset_dir', type=str, default='dataset/rec_dataset_1', help='Recognition dataset directory')
parser.add_argument('--train_data_dir', type=str, default='train_data/meter_detection', help='Training data directory')
parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs to use')
parser.add_argument('--max_det_epochs', type=int, default=200, help='Max detection training epochs')
parser.add_argument('--max_rec_epochs', type=int, default=500, help='Max recognition training epochs')
parser.add_argument('--det_batch_size', type=int, default=8, help='Detection batch size')
parser.add_argument('--rec_batch_size', type=int, default=64, help='Recognition batch size')
parser.add_argument('--det_lr', type=float, default=0.001, help='Detection learning rate')
parser.add_argument('--rec_lr', type=float, default=0.001, help='Recognition learning rate')
args = parser.parse_args()

# Setup MLflow
def setup_mlflow():
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    
    # Set experiment
    mlflow.set_experiment(args.exp_name)
    client = MlflowClient()
    
    # Log system info
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment: {args.exp_name}")
    print(f"GPU available: {paddle.device.is_compiled_with_cuda()}")
    print(f"Paddle version: {paddle.__version__}")
    return client

# Ensure we're in the PaddleOCR root directory
def setup_environment():
    # On Kaggle, get repository code if needed
    if os.path.exists('/kaggle/input/github-paddleocr-training'):
        os.chdir('/kaggle/working')
        if not os.path.exists('paddleocr'):
            print("Copying PaddleOCR code from dataset...")
            subprocess.run('cp -r /kaggle/input/github-paddleocr-training paddleocr', shell=True)
            os.chdir('paddleocr')
    else:
        # Assuming we're in the model_training/notebooks directory
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__"), "../.."))
        os.chdir(ROOT_DIR)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Check if key directories exist
    assert os.path.exists('ppocr'), "Not in PaddleOCR root directory!"
    assert os.path.exists('tools'), "PaddleOCR tools directory not found!"
    
    # Create necessary directories
    os.makedirs(args.det_dataset_dir, exist_ok=True)
    os.makedirs(args.rec_dataset_dir, exist_ok=True)
    os.makedirs(args.train_data_dir, exist_ok=True)

# Prepare training data from Kaggle dataset if needed
def prepare_training_data():
    # On Kaggle, link training data from dataset
    if os.path.exists('/kaggle/input/paddleocr-training-data'):
        print("Linking training data from Kaggle dataset...")
        
        # For detection dataset
        os.makedirs(f"{args.det_dataset_dir}/images", exist_ok=True)
        if not os.listdir(f"{args.det_dataset_dir}/images"):
            subprocess.run(f'ln -sf /kaggle/input/paddleocr-training-data/det_dataset_1/images/* {args.det_dataset_dir}/images/', shell=True)
            subprocess.run(f'cp /kaggle/input/paddleocr-training-data/det_dataset_1/*.txt {args.det_dataset_dir}/', shell=True)
        
        # For recognition dataset
        os.makedirs(f"{args.rec_dataset_dir}/images", exist_ok=True)
        if not os.listdir(f"{args.rec_dataset_dir}/images"):
            subprocess.run(f'ln -sf /kaggle/input/paddleocr-training-data/rec_dataset_1/images/* {args.rec_dataset_dir}/images/', shell=True)
            subprocess.run(f'cp /kaggle/input/paddleocr-training-data/rec_dataset_1/*.txt {args.rec_dataset_dir}/', shell=True)
        
        print("Training data prepared.")

# Check if detection dataset needs conversion
def prepare_detection_data():
    # Data preparation and visualization
    train_label = os.path.join(args.train_data_dir, "train_label.txt")
    test_label = os.path.join(args.train_data_dir, "test_label.txt")

    if not os.path.exists(train_label) or not os.path.exists(test_label):
        print("Training data not found. Preparing from detection dataset.")
        prep_cmd = f"""
        python tools/train_val_split.py \
            --dataset_dir={args.det_dataset_dir} \
            --output_dir={args.train_data_dir} \
            --train_ratio=0.8
        """
        print(f"Running: {prep_cmd}")
        subprocess.run(prep_cmd, shell=True)
        print("Data preparation completed.")
    else:
        print("Training data already prepared.")
        # Show some statistics
        with open(train_label, 'r') as f:
            train_lines = len(f.readlines())
        with open(test_label, 'r') as f:
            test_lines = len(f.readlines())
        print(f"Training samples: {train_lines}")
        print(f"Test samples: {test_lines}")

# Get character dictionary for recognition
def get_character_dict():
    dict_path = "model_training/rec_train/meter_dict.txt"
    
    if not os.path.exists(dict_path):
        print(f"Character dictionary not found: {dict_path}")
        print("Creating a default dictionary with digits and common symbols...")
        
        # Create default dictionary
        default_chars = "0123456789.,-/:"
        os.makedirs(os.path.dirname(dict_path), exist_ok=True)
        
        with open(dict_path, 'w') as f:
            for char in default_chars:
                f.write(f"{char}\n")
        
        print(f"Created default dictionary with {len(default_chars)} characters")
    else:
        # Read existing dictionary
        with open(dict_path, 'r') as f:
            chars = [line.strip() for line in f if line.strip()]
        
        print(f"Character dictionary found with {len(chars)} characters")
    
    return dict_path

# Download pretrained model for detection
def download_det_pretrained_model():
    pretrained_model = "pretrain_models/det_mv3_db_v2.0_train/best_accuracy"
    if not os.path.exists("pretrain_models/det_mv3_db_v2.0_train"):
        print("Downloading pretrained detection model...")
        url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar"
        os.makedirs("pretrain_models", exist_ok=True)
        subprocess.run(f"wget {url} -P pretrain_models/", shell=True)
        subprocess.run(f"tar -xf pretrain_models/det_mv3_db_v2.0_train.tar -C pretrain_models/", shell=True)
    return pretrained_model

# Download pretrained model for recognition
def download_rec_pretrained_model():
    pretrained_model = "pretrain_models/en_PP-OCRv3_rec_train/best_accuracy"
    if not os.path.exists("pretrain_models/en_PP-OCRv3_rec_train"):
        print("Downloading pretrained recognition model...")
        url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar"
        os.makedirs("pretrain_models", exist_ok=True)
        subprocess.run(f"wget {url} -P pretrain_models/", shell=True)
        subprocess.run(f"tar -xf pretrain_models/en_PP-OCRv3_rec_train.tar -C pretrain_models/", shell=True)
    return pretrained_model

# Train detection model with MLflow tracking
def train_detection_model():
    with mlflow.start_run(run_name="detection_training"):
        # Log parameters
        mlflow.log_params({
            "mode": "teacher",
            "gpu_ids": args.gpu_ids,
            "max_epochs": args.max_det_epochs,
            "batch_size": args.det_batch_size,
            "learning_rate": args.det_lr,
            "dataset_dir": args.det_dataset_dir,
            "train_data_dir": args.train_data_dir
        })
        
        # Download pretrained model
        pretrained_model = download_det_pretrained_model()
        
        # Generate training command for teacher model
        teacher_output_dir = "./model_training/det_train/output/meter_teacher"
        os.makedirs(teacher_output_dir, exist_ok=True)
        
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
        
        print("Starting teacher model training...")
        print(f"Command: {teacher_cmd}")
        
        # Execute teacher training
        start_time = time.time()
        process = subprocess.Popen(teacher_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Track training output
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
            # Extract and log metrics from output
            if "avg_acc:" in line:
                try:
                    acc = float(line.split("avg_acc:")[1].split(',')[0].strip())
                    step = int(line.split("[")[1].split("/")[0].strip())
                    mlflow.log_metric("teacher_accuracy", acc, step=step)
                except:
                    pass
            
            if "lr:" in line:
                try:
                    lr = float(line.split("lr:")[1].split(',')[0].strip())
                    step = int(line.split("[")[1].split("/")[0].strip())
                    mlflow.log_metric("teacher_learning_rate", lr, step=step)
                except:
                    pass
        
        process.wait()
        training_time = time.time() - start_time
        mlflow.log_metric("teacher_training_time", training_time)
        
        # Check if teacher model was successfully created
        if not os.path.exists(f"{teacher_output_dir}/best_accuracy.pdparams"):
            print("Teacher model training failed!")
            return False
        
        # Train student model (knowledge distillation)
        student_output_dir = "./model_training/det_train/output/meter_student"
        os.makedirs(student_output_dir, exist_ok=True)
        
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
        
        print("Starting student model training...")
        print(f"Command: {student_cmd}")
        
        # Execute student training
        start_time = time.time()
        process = subprocess.Popen(student_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Track training output
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
            # Extract and log metrics from output
            if "avg_acc:" in line:
                try:
                    acc = float(line.split("avg_acc:")[1].split(',')[0].strip())
                    step = int(line.split("[")[1].split("/")[0].strip())
                    mlflow.log_metric("student_accuracy", acc, step=step)
                except:
                    pass
            
            if "lr:" in line:
                try:
                    lr = float(line.split("lr:")[1].split(',')[0].strip())
                    step = int(line.split("[")[1].split("/")[0].strip())
                    mlflow.log_metric("student_learning_rate", lr, step=step)
                except:
                    pass
        
        process.wait()
        training_time = time.time() - start_time
        mlflow.log_metric("student_training_time", training_time)
        
        # Log artifacts - model files
        if os.path.exists(f"{student_output_dir}/best_accuracy.pdparams"):
            mlflow.log_artifact(f"{student_output_dir}/best_accuracy.pdparams", "models/detection")
            return True
        else:
            print("Student model training failed!")
            return False

# Train recognition model with MLflow tracking
def train_recognition_model():
    with mlflow.start_run(run_name="recognition_training"):
        # Get character dictionary
        dict_path = get_character_dict()
        
        # Log parameters
        mlflow.log_params({
            "mode": "standard",
            "gpu_ids": args.gpu_ids,
            "max_epochs": args.max_rec_epochs,
            "batch_size": args.rec_batch_size,
            "learning_rate": args.rec_lr,
            "dataset_dir": args.rec_dataset_dir,
            "dict_path": dict_path
        })
        
        # Download pretrained model
        pretrained_model = download_rec_pretrained_model()
        
        # Generate training command
        output_dir = "./model_training/rec_train/output/meter_rec"
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if config exists, if not create it
        config_dir = "configs/rec/meter"
        os.makedirs(config_dir, exist_ok=True)
        if not os.path.exists(f"{config_dir}/meter_PP-OCRv3_rec.yml"):
            print("Creating recognition config file...")
            subprocess.run(f"cp configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml {config_dir}/meter_PP-OCRv3_rec.yml", shell=True)
            
            # Modify config for meter recognition
            with open(f"{config_dir}/meter_PP-OCRv3_rec.yml", 'r') as f:
                config = f.read()
            
            # Update data paths and character dict path
            config = config.replace("Train.dataset.data_dir", f"# Train.dataset.data_dir")
            config = config.replace("Eval.dataset.data_dir", f"# Eval.dataset.data_dir")
            
            with open(f"{config_dir}/meter_PP-OCRv3_rec.yml", 'w') as f:
                f.write(config)
        
        # Command for recognition training
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
        
        print("Starting recognition model training...")
        print(f"Command: {cmd}")
        
        # Execute training
        start_time = time.time()
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Track training output
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
            # Extract and log metrics from output
            if "acc:" in line:
                try:
                    acc = float(line.split("acc:")[1].split(',')[0].strip())
                    step = int(line.split("[")[1].split("/")[0].strip())
                    mlflow.log_metric("accuracy", acc, step=step)
                except:
                    pass
            
            if "norm_edit_dis:" in line:
                try:
                    edit_distance = float(line.split("norm_edit_dis:")[1].split(',')[0].strip())
                    step = int(line.split("[")[1].split("/")[0].strip())
                    mlflow.log_metric("edit_distance", edit_distance, step=step)
                except:
                    pass
            
            if "lr:" in line:
                try:
                    lr = float(line.split("lr:")[1].split(',')[0].strip())
                    step = int(line.split("[")[1].split("/")[0].strip())
                    mlflow.log_metric("learning_rate", lr, step=step)
                except:
                    pass
        
        process.wait()
        training_time = time.time() - start_time
        mlflow.log_metric("training_time", training_time)
        
        # Log artifacts - model files
        if os.path.exists(f"{output_dir}/best_accuracy.pdparams"):
            mlflow.log_artifact(f"{output_dir}/best_accuracy.pdparams", "models/recognition")
            mlflow.log_artifact(dict_path, "models/recognition")
            return True
        else:
            print("Recognition model training failed!")
            return False

# Package trained models for download
def package_models():
    print("Packaging trained models for download...")
    model_dir = "./saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Copy detection models
    det_teacher_dir = os.path.join(model_dir, "detection/teacher")
    det_student_dir = os.path.join(model_dir, "detection/student")
    os.makedirs(det_teacher_dir, exist_ok=True)
    os.makedirs(det_student_dir, exist_ok=True)
    
    if os.path.exists("./model_training/det_train/output/meter_teacher/best_accuracy.pdparams"):
        subprocess.run(f"cp ./model_training/det_train/output/meter_teacher/best_accuracy.* {det_teacher_dir}/", shell=True)
    
    if os.path.exists("./model_training/det_train/output/meter_student/best_accuracy.pdparams"):
        subprocess.run(f"cp ./model_training/det_train/output/meter_student/best_accuracy.* {det_student_dir}/", shell=True)
    
    # Copy recognition models
    rec_dir = os.path.join(model_dir, "recognition")
    os.makedirs(rec_dir, exist_ok=True)
    
    if os.path.exists("./model_training/rec_train/output/meter_rec/best_accuracy.pdparams"):
        subprocess.run(f"cp ./model_training/rec_train/output/meter_rec/best_accuracy.* {rec_dir}/", shell=True)
    
    # Copy dictionary
    if os.path.exists("model_training/rec_train/meter_dict.txt"):
        subprocess.run(f"cp model_training/rec_train/meter_dict.txt {rec_dir}/", shell=True)
    
    # Create a zip file for easy download
    subprocess.run(f"zip -r trained_models.zip {model_dir}", shell=True)
    print(f"Models packaged to trained_models.zip")
    
    # In Kaggle, copy to output directory
    if os.path.exists('/kaggle/working'):
        subprocess.run(f"cp trained_models.zip /kaggle/working/", shell=True)
        print(f"Models copied to Kaggle working directory for download")

# Main function
def main():
    print("=" * 50)
    print("PaddleOCR Training Pipeline")
    print("=" * 50)
    
    # Setup environment and MLflow
    setup_environment()
    client = setup_mlflow()
    
    # Prepare dataset
    prepare_training_data()
    
    train_success = True
    
    # Run detection training if requested
    if not args.rec_only:
        print("\n" + "=" * 50)
        print("DETECTION MODEL TRAINING")
        print("=" * 50)
        prepare_detection_data()
        det_success = train_detection_model()
        train_success = train_success and det_success
    
    # Run recognition training if requested
    if not args.det_only:
        print("\n" + "=" * 50)
        print("RECOGNITION MODEL TRAINING")
        print("=" * 50)
        rec_success = train_recognition_model()
        train_success = train_success and rec_success
    
    # Package models for download
    if train_success:
        package_models()
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED WITH ERRORS")
        print("=" * 50)

if __name__ == "__main__":
    main() 