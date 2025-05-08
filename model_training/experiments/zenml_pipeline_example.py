#!/usr/bin/env python
# coding: utf-8

"""
Example ZenML pipeline for PaddleOCR training
This is a starter template showing how to integrate with ZenML
"""

import os
import sys
from typing import Dict, Any, Tuple
import numpy as np

# Import ZenML
from zenml.pipelines import pipeline
from zenml.steps import step
from zenml.client import Client

# Import MLflow
import mlflow
from mlflow.tracking import MlflowClient

# Setup ZenML
def setup_zenml():
    """Initialize ZenML if not already initialized"""
    try:
        client = Client()
        if not client.active_stack:
            print("No active ZenML stack found. Please run 'zenml init' first.")
            sys.exit(1)
        return client
    except Exception as e:
        print(f"Error initializing ZenML: {e}")
        print("Please make sure ZenML is installed and initialized.")
        sys.exit(1)

# Define pipeline steps
@step
def prepare_data(
    det_dataset_dir: str,
    rec_dataset_dir: str,
    train_data_dir: str
) -> Dict[str, Any]:
    """Prepare data for PaddleOCR training"""
    import subprocess
    import os
    
    # Create necessary directories
    os.makedirs(det_dataset_dir, exist_ok=True)
    os.makedirs(rec_dataset_dir, exist_ok=True)
    os.makedirs(train_data_dir, exist_ok=True)
    
    # Check if training data exists
    train_label = os.path.join(train_data_dir, "train_label.txt")
    test_label = os.path.join(train_data_dir, "test_label.txt")
    
    # If training data doesn't exist, prepare it
    if not os.path.exists(train_label) or not os.path.exists(test_label):
        print("Preparing training data...")
        prep_cmd = f"""
        python tools/train_val_split.py \
            --dataset_dir={det_dataset_dir} \
            --output_dir={train_data_dir} \
            --train_ratio=0.8
        """
        subprocess.run(prep_cmd, shell=True)
    
    # Count training samples
    train_count = 0
    test_count = 0
    
    if os.path.exists(train_label):
        with open(train_label, 'r') as f:
            train_count = len(f.readlines())
    
    if os.path.exists(test_label):
        with open(test_label, 'r') as f:
            test_count = len(f.readlines())
    
    data_info = {
        "det_dataset_dir": det_dataset_dir,
        "rec_dataset_dir": rec_dataset_dir,
        "train_data_dir": train_data_dir,
        "train_samples": train_count,
        "test_samples": test_count,
    }
    
    return data_info

@step(enable_cache=False)
def train_detection(
    data_info: Dict[str, Any],
    gpu_ids: str = "0",
    max_epochs: int = 200,
    batch_size: int = 8,
    learning_rate: float = 0.001
) -> str:
    """Train detection model"""
    import subprocess
    import os
    import time
    import mlflow
    
    # Download pretrained model
    pretrained_model = "pretrain_models/det_mv3_db_v2.0_train/best_accuracy"
    if not os.path.exists("pretrain_models/det_mv3_db_v2.0_train"):
        print("Downloading pretrained detection model...")
        url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar"
        os.makedirs("pretrain_models", exist_ok=True)
        subprocess.run(f"wget {url} -P pretrain_models/", shell=True)
        subprocess.run(f"tar -xf pretrain_models/det_mv3_db_v2.0_train.tar -C pretrain_models/", shell=True)
    
    # Setup output directory
    output_dir = "./model_training/det_train/output/meter_teacher"
    os.makedirs(output_dir, exist_ok=True)
    
    # Start MLflow run
    with mlflow.start_run(run_name="detection_training"):
        # Log parameters
        mlflow.log_params({
            "mode": "teacher",
            "gpu_ids": gpu_ids,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples": data_info["train_samples"],
            "test_samples": data_info["test_samples"]
        })
        
        # Command for teacher model training
        cmd = f"""python -m paddle.distributed.launch --gpus='{gpu_ids}' tools/train.py \
            -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_teacher.yml \
            -o Global.pretrained_model={pretrained_model} \
            Global.save_model_dir={output_dir} \
            Train.dataset.data_dir={data_info['train_data_dir']} \
            Train.dataset.label_file_list=['{os.path.join(data_info['train_data_dir'], "train_label.txt")}'] \
            Eval.dataset.data_dir={data_info['train_data_dir']} \
            Eval.dataset.label_file_list=['{os.path.join(data_info['train_data_dir'], "test_label.txt")}'] \
            Train.loader.batch_size_per_card={batch_size} \
            Optimizer.lr.values=[{learning_rate},{learning_rate/10}] \
            Global.epoch_num={max_epochs} \
            Global.save_epoch_step=5 \
            Global.eval_batch_step=[0, 500]"""
        
        print("Starting detection model training...")
        
        # Run training
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Track metrics
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
            # Log metrics from output
            if "avg_acc:" in line:
                try:
                    acc = float(line.split("avg_acc:")[1].split(',')[0].strip())
                    step = int(line.split("[")[1].split("/")[0].strip())
                    mlflow.log_metric("accuracy", acc, step=step)
                except:
                    pass
        
        process.wait()
        
        # Log model artifact
        if os.path.exists(f"{output_dir}/best_accuracy.pdparams"):
            mlflow.log_artifact(f"{output_dir}/best_accuracy.pdparams", "models")
            return os.path.join(output_dir, "best_accuracy")
        else:
            return ""

@step(enable_cache=False)
def train_recognition(
    data_info: Dict[str, Any],
    gpu_ids: str = "0",
    max_epochs: int = 500,
    batch_size: int = 64,
    learning_rate: float = 0.001
) -> str:
    """Train recognition model"""
    import subprocess
    import os
    import time
    import mlflow
    
    # Create character dictionary
    dict_path = "model_training/rec_train/meter_dict.txt"
    
    if not os.path.exists(dict_path):
        print("Creating character dictionary...")
        default_chars = "0123456789.,-/:"
        os.makedirs(os.path.dirname(dict_path), exist_ok=True)
        
        with open(dict_path, 'w') as f:
            for char in default_chars:
                f.write(f"{char}\n")
    
    # Download pretrained model
    pretrained_model = "pretrain_models/en_PP-OCRv3_rec_train/best_accuracy"
    if not os.path.exists("pretrain_models/en_PP-OCRv3_rec_train"):
        print("Downloading pretrained recognition model...")
        url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar"
        os.makedirs("pretrain_models", exist_ok=True)
        subprocess.run(f"wget {url} -P pretrain_models/", shell=True)
        subprocess.run(f"tar -xf pretrain_models/en_PP-OCRv3_rec_train.tar -C pretrain_models/", shell=True)
    
    # Setup output directory
    output_dir = "./model_training/rec_train/output/meter_rec"
    os.makedirs(output_dir, exist_ok=True)
    
    # Start MLflow run
    with mlflow.start_run(run_name="recognition_training"):
        # Log parameters
        mlflow.log_params({
            "mode": "standard",
            "gpu_ids": gpu_ids,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        })
        
        # Check/create config directory
        config_dir = "configs/rec/meter"
        os.makedirs(config_dir, exist_ok=True)
        if not os.path.exists(f"{config_dir}/meter_PP-OCRv3_rec.yml"):
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
        cmd = f"""python -m paddle.distributed.launch --gpus='{gpu_ids}' tools/train.py \
            -c configs/rec/meter/meter_PP-OCRv3_rec.yml \
            -o Global.pretrained_model={pretrained_model} \
            Global.character_dict_path={dict_path} \
            Global.save_model_dir={output_dir} \
            Train.dataset.data_dir={data_info['rec_dataset_dir']} \
            Train.dataset.label_file_list=['{os.path.join(data_info['rec_dataset_dir'], "train_list.txt")}'] \
            Eval.dataset.data_dir={data_info['rec_dataset_dir']} \
            Eval.dataset.label_file_list=['{os.path.join(data_info['rec_dataset_dir'], "test_list.txt")}'] \
            Train.loader.batch_size_per_card={batch_size} \
            Optimizer.lr.values=[{learning_rate},{learning_rate/10}] \
            Global.epoch_num={max_epochs} \
            Global.save_epoch_step=5 \
            Global.eval_batch_step=[0, 500]"""
        
        print("Starting recognition model training...")
        
        # Run training
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Track metrics
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
            # Log metrics from output
            if "acc:" in line:
                try:
                    acc = float(line.split("acc:")[1].split(',')[0].strip())
                    step = int(line.split("[")[1].split("/")[0].strip())
                    mlflow.log_metric("accuracy", acc, step=step)
                except:
                    pass
        
        process.wait()
        
        # Log model artifact
        if os.path.exists(f"{output_dir}/best_accuracy.pdparams"):
            mlflow.log_artifact(f"{output_dir}/best_accuracy.pdparams", "models")
            mlflow.log_artifact(dict_path, "models")
            return os.path.join(output_dir, "best_accuracy")
        else:
            return ""

@step
def package_models(
    det_model_path: str, 
    rec_model_path: str
) -> str:
    """Package trained models for deployment"""
    import subprocess
    import os
    
    if not det_model_path or not rec_model_path:
        print("One or both models failed to train. Skipping packaging.")
        return ""
    
    print("Packaging models...")
    
    # Create directories
    model_dir = "./packaged_models"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "detection"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "recognition"), exist_ok=True)
    
    # Copy detection model
    if os.path.exists(f"{det_model_path}.pdparams"):
        subprocess.run(f"cp {det_model_path}.* {model_dir}/detection/", shell=True)
    
    # Copy recognition model
    if os.path.exists(f"{rec_model_path}.pdparams"):
        subprocess.run(f"cp {rec_model_path}.* {model_dir}/recognition/", shell=True)
    
    # Copy dictionary
    dict_path = "model_training/rec_train/meter_dict.txt"
    if os.path.exists(dict_path):
        subprocess.run(f"cp {dict_path} {model_dir}/recognition/", shell=True)
    
    # Create zip file
    subprocess.run(f"zip -r packaged_models.zip {model_dir}", shell=True)
    
    return "packaged_models.zip"

# Define the full pipeline
@pipeline
def paddleocr_training_pipeline(
    det_dataset_dir: str = "dataset/det_dataset_1",
    rec_dataset_dir: str = "dataset/rec_dataset_1",
    train_data_dir: str = "train_data/meter_detection",
    gpu_ids: str = "0",
    max_det_epochs: int = 50,
    max_rec_epochs: int = 100,
    det_batch_size: int = 8,
    rec_batch_size: int = 64,
    det_lr: float = 0.001,
    rec_lr: float = 0.001
):
    """Full PaddleOCR training pipeline"""
    # Prepare data
    data_info = prepare_data(
        det_dataset_dir=det_dataset_dir,
        rec_dataset_dir=rec_dataset_dir,
        train_data_dir=train_data_dir
    )
    
    # Train detection model
    det_model_path = train_detection(
        data_info=data_info,
        gpu_ids=gpu_ids,
        max_epochs=max_det_epochs,
        batch_size=det_batch_size,
        learning_rate=det_lr
    )
    
    # Train recognition model
    rec_model_path = train_recognition(
        data_info=data_info,
        gpu_ids=gpu_ids,
        max_epochs=max_rec_epochs,
        batch_size=rec_batch_size,
        learning_rate=rec_lr
    )
    
    # Package models
    package_path = package_models(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path
    )
    
    return package_path

# Main function to run the pipeline
def main():
    """Run the ZenML pipeline"""
    # Set up ZenML client
    client = setup_zenml()
    print(f"Using ZenML stack: {client.active_stack.name}")
    
    # Run the pipeline
    try:
        pipeline_instance = paddleocr_training_pipeline(
            det_dataset_dir="dataset/det_dataset_1",
            rec_dataset_dir="dataset/rec_dataset_1",
            train_data_dir="train_data/meter_detection",
            max_det_epochs=10,  # Low value for testing
            max_rec_epochs=10,  # Low value for testing
            gpu_ids="0"
        )
        pipeline_instance.run()
        print("Pipeline run completed successfully!")
    except Exception as e:
        print(f"Pipeline run failed: {e}")

if __name__ == "__main__":
    main() 