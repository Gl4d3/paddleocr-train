# PaddleOCR Model Training

This directory contains the organized training code and configuration files for training OCR models using PaddleOCR, with support for MLflow tracking and ZenML pipeline integration.

## Directory Structure

```
model_training/
├── configs/            # Configuration files
├── scripts/            # Utility scripts
│   ├── dataset/        # Dataset preparation scripts
│   ├── training/       # Training scripts
│   └── deployment/     # Deployment scripts
├── notebooks/          # Interactive notebooks
├── experiments/        # Experiment tracking
│   ├── mlflow/         # MLflow related files
│   └── metrics/        # Saved metrics
├── det_train/          # Detection model specific files
│   ├── configs/        # Configuration files for detection models
│   ├── output/         # Training output directory
│   │   ├── meter_teacher/ # Teacher model checkpoints and logs
│   │   └── meter_student/ # Student model checkpoints and logs
├── rec_train/          # Recognition model specific files
    ├── configs/        # Configuration files for recognition models
    ├── meter_dict.txt  # Character dictionary for meter readings
    ├── output/         # Training output directory
```

## Training Approaches

There are multiple ways to train the OCR models:

### 1. Local Training using Scripts

The most direct approach using the training scripts:

```bash
# For detection model training
cd PaddleOCR
python model_training/scripts/training/kaggle_training.py --det_only

# For recognition model training
python model_training/scripts/training/kaggle_training.py --rec_only

# For full pipeline training
python model_training/scripts/training/kaggle_training.py
```

### 2. Kaggle Training (Cloud GPUs)

For training on Kaggle's GPU:

1. Create a Kaggle notebook using the template:
   ```python
   # Use the content from model_training/scripts/training/kaggle_notebook_training.py as a template
   ```

2. Upload your training data to Kaggle as a dataset or download it within the notebook
3. Run the notebook on Kaggle with GPU enabled

#### API-Triggered Kaggle Training

You can also trigger training via the Kaggle API:

```bash
# Install Kaggle API
pip install kaggle

# Configure your API key
# Place kaggle.json in ~/.kaggle/ directory
chmod 600 ~/.kaggle/kaggle.json

# Trigger training
python model_training/scripts/training/kaggle_api_trigger.py --notebook path/to/notebook.ipynb --gpu
```

### 3. ZenML Pipeline Training (Experiment Tracking)

For a more structured pipeline approach with tracking:

```bash
# Set up ZenML with MLflow
python model_training/experiments/setup_zenml.py

# Run the training pipeline
python model_training/experiments/zenml_pipeline_example.py
```

## Experiment Tracking with MLflow

All training approaches include MLflow integration for tracking experiments:

### Starting MLflow Server

```bash
# Set up a local MLflow server
python model_training/experiments/mlflow/setup_mlflow.py

# View the MLflow UI (in a separate terminal)
mlflow ui --host 0.0.0.0 --port 5000
```

Visit `http://localhost:5000` to view the MLflow UI with your training metrics.

### Tracked Metrics

- Detection model:
  - Accuracy
  - Learning rate
  - Training time

- Recognition model:
  - Accuracy
  - Edit distance
  - Learning rate
  - Training time

## Model Knowledge Distillation

The training uses a knowledge distillation approach:

1. **Teacher Model**: ResNet50 backbone with LKPAN neck and DBHead
2. **Student Model**: MobileNetV3 backbone with RSEFPN neck and DBHead

This results in a deployment-friendly model that maintains high accuracy.

## Configuring Training

Key parameters can be adjusted for both detection and recognition training:

- GPU IDs
- Number of epochs
- Batch size
- Learning rate
- Dataset paths

See the argument documentation in each script for details.

## Adding Your Own Dataset

1. Create directories for your dataset:
   ```bash
   mkdir -p dataset/custom_det_dataset/images
   mkdir -p dataset/custom_rec_dataset/images
   ```

2. Prepare annotation files in the required format:
   - Detection: Ground truth polygons
   - Recognition: Text labels

3. Update training scripts to use your dataset paths:
   ```python
   --det_dataset_dir=dataset/custom_det_dataset
   --rec_dataset_dir=dataset/custom_rec_dataset
   ```

## Best Practices

1. **Start small**: Begin with a small subset of data to verify the pipeline works
2. **Monitor training**: Use MLflow to track the training process
3. **Checkpoints**: Save models at regular intervals
4. **Evaluation**: Always evaluate models on a separate test set
5. **Export**: Export trained models to ONNX format for deployment

## Troubleshooting

- **Memory errors**: Reduce batch size or image size
- **Slow training**: Verify GPU usage with `nvidia-smi`
- **Poor accuracy**: Try adjusting learning rate or increasing training data
- **MLflow connection issues**: Verify the tracking server is running 