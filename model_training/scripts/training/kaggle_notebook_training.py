#!/usr/bin/env python
# coding: utf-8

"""
PaddleOCR Training on Kaggle
This file contains cell markers to help create a notebook in the Kaggle UI.
"""

# Cell 1: Check GPU Availability
"""
# Check if GPU is available
"""
# !nvidia-smi

# Cell 2: Install Required Packages
"""
# Install required packages
"""
# !pip install -q paddlepaddle-gpu==2.4.2
# !pip install -q mlflow paddleocr visualdl opencv-python lmdb imgaug pyclipper scikit-image

# Cell 3: Clone Repository
"""
# Clone the repository directly instead of using dataset
# This is simpler than creating a dataset from the repo
"""
import os

# Check if repository already exists
if not os.path.exists('paddleocr-train'):
    # !git clone https://github.com/Gl4d3/paddleocr-train.git
    # %cd paddleocr-train
    pass
else:
    # %cd paddleocr-train
    # Pull latest changes
    # !git pull
    pass

# Check directory structure
# !ls -la

# Cell 4: Set Up MLflow Tracking
"""
# Set up MLflow for experiment tracking
"""
import os
import mlflow
from mlflow.tracking import MlflowClient

# Set up MLflow locally (you can change this to a remote server if needed)
os.makedirs('mlruns', exist_ok=True)
# mlflow.set_tracking_uri('file://$(pwd)/mlruns')
# os.environ['MLFLOW_TRACKING_URI'] = 'file://$(pwd)/mlruns'

# Create the experiment
experiment_name = 'paddleocr_training'
mlflow.set_experiment(experiment_name)

print(f"MLflow tracking at: {mlflow.get_tracking_uri()}")
print(f"MLflow experiment: {experiment_name}")

# Cell 5: Prepare Training Data
"""
# Check and prepare training data
"""
# Check if training data is available
# !mkdir -p dataset/det_dataset_1/images dataset/rec_dataset_1/images train_data/meter_detection

# If you have uploaded data to Kaggle datasets, use this:
if os.path.exists('/kaggle/input/paddleocr-training-data'):
    print("Found training data dataset")
    # !ln -sf /kaggle/input/paddleocr-training-data/det_dataset_1/images/* dataset/det_dataset_1/images/
    # !cp /kaggle/input/paddleocr-training-data/det_dataset_1/*.txt dataset/det_dataset_1/
    
    # !ln -sf /kaggle/input/paddleocr-training-data/rec_dataset_1/images/* dataset/rec_dataset_1/images/
    # !cp /kaggle/input/paddleocr-training-data/rec_dataset_1/*.txt dataset/rec_dataset_1/
else:
    print("No uploaded dataset found. You need to add your own training data.")
    # You could download sample data here if needed

# Cell 6: Run Training
"""
# Run the training pipeline
"""
# Set training parameters
det_dataset_dir = 'dataset/det_dataset_1'
rec_dataset_dir = 'dataset/rec_dataset_1'
train_data_dir = 'train_data/meter_detection'
max_det_epochs = 50  # Reduced for testing, increase for production
max_rec_epochs = 100  # Reduced for testing, increase for production

# Define command
cmd = """python model_training/scripts/training/kaggle_training.py \\
    --exp_name='paddleocr_training' \\
    --tracking_uri='file://$(pwd)/mlruns' \\
    --det_dataset_dir='{det_dataset_dir}' \\
    --rec_dataset_dir='{rec_dataset_dir}' \\
    --train_data_dir='{train_data_dir}' \\
    --gpu_ids='0' \\
    --max_det_epochs={max_det_epochs} \\
    --max_rec_epochs={max_rec_epochs} \\
    --det_batch_size=8 \\
    --rec_batch_size=64"""

print(f"Training command:\n{cmd}")

# Execute the training
# !$cmd

# Cell 7: Package Results for Download
"""
# Package trained models and metrics for download
"""
# !mkdir -p saved_results

# Copy trained models
# !cp -r model_training/det_train/output saved_results/detection_models
# !cp -r model_training/rec_train/output saved_results/recognition_models

# Package MLflow data
# !cp -r mlruns saved_results/mlflow_logs

# Create zip archives for easy download
# !zip -r trained_models.zip saved_results/detection_models saved_results/recognition_models
# !zip -r mlflow_logs.zip saved_results/mlflow_logs

print("Training artifacts ready for download:")
print(" - trained_models.zip - Trained detection and recognition models")
print(" - mlflow_logs.zip - MLflow logs and metrics")

# Cell 8: View Training Results and Metrics
"""
# Display training results and metrics
"""
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("paddleocr_training")
if experiment:
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    for run in runs:
        print(f"Run ID: {run.info.run_id}")
        print(f"Status: {run.info.status}")
        print("Parameters:")
        for k, v in run.data.params.items():
            print(f"  {k}: {v}")
        print("Metrics:")
        for k, v in run.data.metrics.items():
            print(f"  {k}: {v}")
        print("====================================")
else:
    print("No experiment found. Training may not have completed successfully.") 