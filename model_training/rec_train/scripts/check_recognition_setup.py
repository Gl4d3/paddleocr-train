#!/usr/bin/env python3
"""
Script to check if the recognition model training setup is correct.
This verifies all paths, configurations, and data formats required for training.
"""

import os
import sys
import argparse
from pathlib import Path

def check_file_exists(file_path, message=None, critical=False):
    """Check if a file exists and print a message"""
    file_exists = os.path.isfile(file_path)
    status = "✓" if file_exists else "✗"
    if message is None:
        message = f"Checking {file_path}"
    
    print(f"[{status}] {message}")
    
    if critical and not file_exists:
        print(f"ERROR: Critical file {file_path} not found!")
        return False
    return file_exists

def check_dir_exists(dir_path, message=None, critical=False, create=False):
    """Check if a directory exists and optionally create it"""
    if create and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"[+] Created directory {dir_path}")
    
    dir_exists = os.path.isdir(dir_path)
    status = "✓" if dir_exists else "✗"
    if message is None:
        message = f"Checking {dir_path}"
    
    print(f"[{status}] {message}")
    
    if critical and not dir_exists:
        print(f"ERROR: Critical directory {dir_path} not found!")
        return False
    return dir_exists

def count_lines(file_path):
    """Count the number of lines in a file"""
    if not os.path.isfile(file_path):
        return 0
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

def check_recognition_setup():
    """Check all components of recognition model training setup"""
    # Get the PaddleOCR root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    os.chdir(root_dir)
    print(f"PaddleOCR root directory: {root_dir}")
    
    # Check critical directories and files
    print("\n== Checking Directory Structure ==")
    check_dir_exists("configs/rec/meter", "Config directory for meter recognition", True, True)
    check_dir_exists("model_training/rec_train/output", "Output directory for recognition models", True, True)
    check_dir_exists("dataset/rec_dataset_1/images", "Recognition dataset images directory", True, True)
    
    # Check configuration files
    print("\n== Checking Configuration Files ==")
    config_paths = {
        "Standard config": "model_training/rec_train/configs/meter_PP-OCRv3_rec.yml",
        "Distillation config": "model_training/rec_train/configs/meter_PP-OCRv3_rec_distillation.yml"
    }
    
    configs_ok = True
    for name, path in config_paths.items():
        if not check_file_exists(path, f"{name} exists"):
            configs_ok = False
    
    # If configs exist, copy them to the main config directory
    if configs_ok:
        os.system("cp model_training/rec_train/configs/meter_PP-OCRv3_rec.yml configs/rec/meter/")
        os.system("cp model_training/rec_train/configs/meter_PP-OCRv3_rec_distillation.yml configs/rec/meter/")
        print("[+] Copied configuration files to configs/rec/meter/")
    
    # Check character dictionary
    print("\n== Checking Character Dictionary ==")
    dict_path = "model_training/rec_train/meter_dict.txt"
    if check_file_exists(dict_path, "Character dictionary exists", True):
        # Count the number of characters in the dictionary
        with open(dict_path, 'r') as f:
            chars = [line.strip() for line in f if line.strip()]
        print(f"    - Dictionary has {len(chars)} characters: {', '.join(chars)}")
    
    # Check dataset files
    print("\n== Checking Recognition Dataset ==")
    train_path = "dataset/rec_dataset_1/train_list.txt"
    test_path = "dataset/rec_dataset_1/test_list.txt"
    
    train_lines = count_lines(train_path)
    test_lines = count_lines(test_path)
    
    check_file_exists(train_path, f"Train list exists with {train_lines} samples", True)
    check_file_exists(test_path, f"Test list exists with {test_lines} samples", True)
    
    # If dataset files exist but are empty, inform the user
    if train_lines == 0 or test_lines == 0:
        print("\nWARNING: Dataset files are empty. You need to extract text regions first.")
        print("Run the extraction script:")
        print("python3 model_training/rec_train/scripts/extract_meter_readings.py \\")
        print("  --det_data_dir=dataset/det_dataset_1 \\")
        print("  --rec_output_dir=dataset/rec_dataset_1 \\")
        print("  --train_data=train_data/meter_detection/train_label.txt \\")
        print("  --test_data=train_data/meter_detection/test_label.txt")
    
    # Check for placeholder labels
    if train_lines > 0:
        with open(train_path, 'r') as f:
            first_line = f.readline().strip()
            if '###' in first_line:
                print("\nWARNING: Train list contains placeholder labels (###).")
                print("You need to replace these with actual meter readings.")
    
    # Check pretrained models
    print("\n== Checking Pretrained Models ==")
    pretrained_dir = "pretrain_models/en_PP-OCRv3_rec_train"
    pretrained_tar = "pretrain_models/en_PP-OCRv3_rec_train.tar"
    
    if check_dir_exists(pretrained_dir, "Pretrained model directory exists"):
        check_file_exists(f"{pretrained_dir}/best_accuracy.pdparams", "Pretrained model weights exist")
    elif check_file_exists(pretrained_tar, "Pretrained model archive exists"):
        print("[i] Pretrained model archive needs to be extracted")
        print("    Run: tar -xf pretrain_models/en_PP-OCRv3_rec_train.tar -C pretrain_models/")
    else:
        print("[i] Pretrained model not found. It will be downloaded during training.")
    
    # Summary
    print("\n== Recognition Training Setup Summary ==")
    if configs_ok and check_file_exists(dict_path) and train_lines > 0 and test_lines > 0:
        print("✓ Basic setup is complete. You can run the training script:")
        print("    ./model_training/rec_train/scripts/train_meter_recognition.sh")
    else:
        print("✗ Setup is incomplete. Please address the issues above before training.")

if __name__ == "__main__":
    check_recognition_setup() 