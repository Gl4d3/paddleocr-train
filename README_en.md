English | [简体中文](README.md)

# PaddleOCR Customized Training Pipeline

This repository contains a customized implementation of PaddleOCR with enhanced training workflows for OCR model development. The main focus is on providing structured training pipelines with better experiment tracking and cloud GPU integration.

## Key Features

- **Organized Training Pipelines** with MLflow experiment tracking
- **Kaggle GPU Integration** for training on free cloud GPUs
- **ZenML Workflow Management** for reproducible training pipelines
- **Multiple Training Approaches** to suit different needs and environments

## Directory Structure

```
.
├── PaddleOCR/           # Original PaddleOCR code
├── model_training/      # Custom training pipelines
│   ├── configs/         # Configuration files
│   ├── scripts/         # Training, dataset and deployment scripts
│   ├── experiments/     # MLflow and ZenML integration
│   ├── det_train/       # Detection model training
│   └── rec_train/       # Recognition model training
```

## Getting Started

### 1. Local Training

```bash
# For detection model training
python model_training/scripts/training/kaggle_training.py --det_only

# For recognition model training
python model_training/scripts/training/kaggle_training.py --rec_only

# For full pipeline training
python model_training/scripts/training/kaggle_training.py
```

### 2. Kaggle GPU Training

Use the provided notebook template:
```bash
# Create a Kaggle notebook using the template
python model_training/scripts/training/kaggle_notebook_training.py
```

### 3. API-Triggered Kaggle Training

```bash
# Configure Kaggle API access first
python model_training/scripts/training/kaggle_api_trigger.py --notebook path/to/notebook.ipynb --gpu
```

### 4. ZenML Structured Pipelines

```bash
# Set up ZenML with MLflow
python model_training/experiments/setup_zenml.py

# Run the training pipeline
python model_training/experiments/zenml_pipeline_example.py
```

## Experiment Tracking with MLflow

```bash
# Start MLflow server
python model_training/experiments/mlflow/setup_mlflow.py

# View UI (in separate terminal)
mlflow ui --host 0.0.0.0 --port 5000
```

## Documentation

For detailed documentation on the training approaches, see:
- [Model Training README](./model_training/README.md) - Complete training documentation
- [Kaggle Integration](./model_training/scripts/training/kaggle_notebook_training.py) - Kaggle notebook templates
- [ZenML Pipelines](./model_training/experiments/zenml_pipeline_example.py) - Structured ML pipeline examples

## Acknowledgements

This repository builds upon the excellent work of the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) team.

For the original PaddleOCR documentation, see [PaddleOCR documentation](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/README_en.md). 