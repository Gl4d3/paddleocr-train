# PaddleOCR Training Notebooks

Jupyter notebooks for training custom OCR models with PaddleOCR.

## Notebooks

- **detection_training.ipynb**: Train text detection models
- **recognition_training.ipynb**: Train text recognition models  
- **full_ocr_pipeline.ipynb**: End-to-end OCR pipeline with both models

## Usage

These notebooks are designed to be run from the PaddleOCR root directory. They include:

- Dataset configuration and preparation
- Model training and fine-tuning
- Evaluation and inference examples

## Quick Start

1. Run detection training notebook first
2. Then run recognition training notebook
3. Finally use the pipeline notebook to test the complete system

## Customization

Easily adapt to your dataset by modifying the path variables:

```python
DET_DATASET_DIR = "path/to/detection/dataset"
REC_DATASET_DIR = "path/to/recognition/dataset"
```

## Notes

- Most execution cells are commented out - uncomment to run
- Adjust batch size and training parameters based on your hardware
- The notebooks automatically handle model downloading when needed