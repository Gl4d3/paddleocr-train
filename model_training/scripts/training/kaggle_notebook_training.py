#!/usr/bin/env python
# coding: utf-8

"""
PaddleOCR Training on Kaggle
This file contains cell markers to help create a notebook in the Kaggle UI.
Copy each cell's content into your Kaggle notebook.
"""

# Cell 1: Install Dependencies
"""
# Install required packages
"""
# !pip install -q -r /kaggle/input/paddleocr-training-data/requirements.txt
# !pip install -q paddlepaddle-gpu==2.4.2 paddlepaddle==2.4.2 paddleocr>=2.6.1.3
# !pip install -q mlflow>=2.3.0 visualdl>=2.5.0 opencv-python>=4.6.0 opencv-contrib-python>=4.6.0
# !pip install -q lmdb>=1.3.0 imgaug>=0.4.0 pyclipper>=1.2.1 scikit-image>=0.19.0

# Cell 2: Setup Environment and Check GPU
"""
# Setup environment and check GPU availability
"""
import os
import sys
from pathlib import Path

# Ensure we're in the Kaggle working directory
os.chdir('/kaggle/working')

# Check GPU availability
import paddle
print(f"GPU available: {paddle.device.is_compiled_with_cuda()}")
print(f"Paddle version: {paddle.__version__}")

# Define paths
WORKING_DIR = Path('/kaggle/working')
REPO_DIR = WORKING_DIR / 'paddleocr-train'
DATASET_DIR = WORKING_DIR / 'dataset'
TRAIN_DATA_DIR = WORKING_DIR / 'train_data'
MLRUNS_DIR = WORKING_DIR / 'mlruns'

# Create necessary directories
for dir_path in [REPO_DIR, DATASET_DIR, TRAIN_DATA_DIR, MLRUNS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

print(f"Working directory: {WORKING_DIR}")
print(f"Repository directory: {REPO_DIR}")
print(f"Dataset directory: {DATASET_DIR}")
print(f"Training data directory: {TRAIN_DATA_DIR}")
print(f"MLflow directory: {MLRUNS_DIR}")

# Cell 3: Clone Repository
"""
# Clone the repository
"""
# Check if repository already exists
if not os.path.exists(str(REPO_DIR)):
    print(f"Cloning repository to {REPO_DIR}")
    # !git clone https://github.com/Gl4d3/paddleocr-train.git {REPO_DIR}
else:
    print(f"Repository exists at {REPO_DIR}")
    os.chdir(str(REPO_DIR))
    # !git pull

# Check directory structure
# !ls -la

# Cell 4: Set Up MLflow
"""
# Set up MLflow tracking
"""
import mlflow
from mlflow.tracking import MlflowClient

# Set up MLflow with absolute path
mlflow_tracking_uri = f"file://{MLRUNS_DIR}"
mlflow.set_tracking_uri(mlflow_tracking_uri)
os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri

# Create experiment
experiment_name = 'paddleocr_training'
mlflow.set_experiment(experiment_name)

print(f"MLflow tracking at: {mlflow.get_tracking_uri()}")
print(f"MLflow experiment: {experiment_name}")

# Cell 5: Prepare Training Data
"""
# Prepare training data
"""
# Create dataset directories
det_dataset = DATASET_DIR / 'det_dataset_1'
rec_dataset = DATASET_DIR / 'rec_dataset_1'
train_data = TRAIN_DATA_DIR / 'meter_detection'

for dir_path in [det_dataset, rec_dataset, train_data]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Link data from Kaggle dataset if available
if os.path.exists('/kaggle/input/paddleocr-training-data'):
    print("Found training data dataset")
    # !ln -sf /kaggle/input/paddleocr-training-data/det_dataset_1/images/* {det_dataset}/images/
    # !cp /kaggle/input/paddleocr-training-data/det_dataset_1/*.txt {det_dataset}/
    # !ln -sf /kaggle/input/paddleocr-training-data/rec_dataset_1/images/* {rec_dataset}/images/
    # !cp /kaggle/input/paddleocr-training-data/rec_dataset_1/*.txt {rec_dataset}/
else:
    print("No uploaded dataset found. You need to add your own training data.")

# Cell 6: Run Training
"""
# Run training pipeline
"""
# Training parameters
max_det_epochs = 50
max_rec_epochs = 100

# Define command with absolute paths
cmd = f"""python {REPO_DIR}/model_training/scripts/training/kaggle_training.py \\
    --exp_name='paddleocr_training' \\
    --tracking_uri='{mlflow_tracking_uri}' \\
    --det_dataset_dir='{det_dataset}' \\
    --rec_dataset_dir='{rec_dataset}' \\
    --train_data_dir='{train_data}' \\
    --gpu_ids='0' \\
    --max_det_epochs={max_det_epochs} \\
    --max_rec_epochs={max_rec_epochs} \\
    --det_batch_size=8 \\
    --rec_batch_size=64"""

print(f"Training command:\n{cmd}")

# Execute the training
# !$cmd

# Cell 7: Package Results
"""
# Package training results
"""
# Create results directory
results_dir = WORKING_DIR / 'saved_results'
results_dir.mkdir(exist_ok=True)

# Copy trained models
# !cp -r {REPO_DIR}/model_training/det_train/output {results_dir}/detection_models
# !cp -r {REPO_DIR}/model_training/rec_train/output {results_dir}/recognition_models

# Package MLflow data
# !cp -r {MLRUNS_DIR} {results_dir}/mlflow_logs

# Create zip archives
# !zip -r {WORKING_DIR}/trained_models.zip {results_dir}/detection_models {results_dir}/recognition_models
# !zip -r {WORKING_DIR}/mlflow_logs.zip {results_dir}/mlflow_logs

print("Training artifacts ready for download:")
print(f" - {WORKING_DIR}/trained_models.zip - Trained models")
print(f" - {WORKING_DIR}/mlflow_logs.zip - MLflow logs")

# Cell 8: View Results
"""
# Display training results
"""
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