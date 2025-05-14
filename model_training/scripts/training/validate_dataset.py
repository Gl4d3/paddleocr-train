#!/usr/bin/env python
# coding: utf-8

"""
Dataset Validation Script for PaddleOCR Training
- Validates the existence and format of datasets
- Does NOT modify any data
- Provides detailed reports of any issues found
"""

import os
import sys
import json
import glob
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dataset_validation.log")
    ]
)
logger = logging.getLogger("dataset-validation")

# Parse arguments
parser = argparse.ArgumentParser(description='Validate datasets for PaddleOCR training')
parser.add_argument('--det_dataset_dir', type=str, default='dataset/det_dataset_1', help='Detection dataset directory')
parser.add_argument('--rec_dataset_dir', type=str, default='dataset/rec_dataset_1', help='Recognition dataset directory')
parser.add_argument('--train_data_dir', type=str, default='train_data/meter_detection', help='Training data directory')
parser.add_argument('--check_images', action='store_true', help='Validate image files (may be slow)')
args = parser.parse_args()

def setup_environment():
    """Setup the environment and ensure correct directory structure."""
    # Get the absolute path of the current script
    current_script = os.path.abspath(__file__)
    
    # Navigate to PaddleOCR root
    root_dir = os.path.abspath(os.path.join(os.path.dirname(current_script), "../../.."))
    os.chdir(root_dir)
    
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Verify essential directories exist
    required_dirs = ['ppocr', 'tools']
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    if missing_dirs:
        raise RuntimeError(f"Missing required directories: {', '.join(missing_dirs)}")

def validate_image_file(image_path):
    """Check if an image file is valid and can be opened."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "File exists but cannot be read as an image"
        return True, ""
    except Exception as e:
        return False, str(e)

def validate_detection_dataset():
    """Validate detection dataset structure and format."""
    logger.info(f"Validating detection dataset: {args.det_dataset_dir}")
    
    issues_found = 0
    warnings_found = 0
    
    # Check directory structure
    if not os.path.exists(args.det_dataset_dir):
        logger.error(f"Detection dataset directory not found: {args.det_dataset_dir}")
        return False
    
    images_dir = os.path.join(args.det_dataset_dir, "images")
    if not os.path.exists(images_dir):
        logger.error(f"Images directory not found: {images_dir}")
        issues_found += 1
    
    # Check annotation files
    train_anno = os.path.join(args.det_dataset_dir, "train_verified.txt")
    test_anno = os.path.join(args.det_dataset_dir, "test_verified.txt")
    
    if not os.path.exists(train_anno):
        logger.error(f"Training annotation file not found: {train_anno}")
        issues_found += 1
    
    if not os.path.exists(test_anno):
        logger.error(f"Test annotation file not found: {test_anno}")
        issues_found += 1
    
    # Validate annotation format and image existence
    annotation_files = [('train', train_anno), ('test', test_anno)]
    for anno_type, anno_file in annotation_files:
        if not os.path.exists(anno_file):
            continue
        
        logger.info(f"Validating {anno_type} annotations: {anno_file}")
        with open(anno_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        logger.info(f"Found {len(lines)} entries in {anno_type} annotations")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                logger.warning(f"Empty line in {anno_file} at line {i+1}")
                warnings_found += 1
                continue
                
            try:
                parts = line.split('\t')
                if len(parts) != 2:
                    logger.error(f"Invalid format in {anno_file} at line {i+1}: expected 2 tab-separated parts, got {len(parts)}")
                    issues_found += 1
                    continue
                
                img_path, anno_text = parts
                
                # Check if image exists
                full_img_path = os.path.join(args.det_dataset_dir, img_path)
                if not os.path.exists(full_img_path):
                    logger.error(f"Image not found: {full_img_path} (referenced in {anno_file} line {i+1})")
                    issues_found += 1
                elif args.check_images:
                    valid, error = validate_image_file(full_img_path)
                    if not valid:
                        logger.error(f"Invalid image: {full_img_path} - {error}")
                        issues_found += 1
                
                # Check annotation format (should be JSON)
                try:
                    annotations = json.loads(anno_text)
                    if not isinstance(annotations, list):
                        logger.error(f"Invalid JSON format in {anno_file} line {i+1}: expected a list")
                        issues_found += 1
                        continue
                    
                    for j, ann in enumerate(annotations):
                        if not isinstance(ann, dict):
                            logger.error(f"Invalid annotation object in {anno_file} line {i+1}, item {j}: expected a dictionary")
                            issues_found += 1
                            continue
                        
                        if 'points' not in ann:
                            logger.error(f"Missing 'points' key in {anno_file} line {i+1}, item {j}")
                            issues_found += 1
                        elif not isinstance(ann['points'], list) or len(ann['points']) != 4:
                            logger.error(f"Invalid 'points' format in {anno_file} line {i+1}, item {j}: expected list of 4 points")
                            issues_found += 1
                        
                        if 'transcription' not in ann:
                            logger.warning(f"Missing 'transcription' key in {anno_file} line {i+1}, item {j}")
                            warnings_found += 1
                            
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in {anno_file} line {i+1}")
                    issues_found += 1
            except Exception as e:
                logger.error(f"Error processing {anno_file} line {i+1}: {str(e)}")
                issues_found += 1
    
    # Check prepared training data
    train_label = os.path.join(args.train_data_dir, "train_label.txt")
    test_label = os.path.join(args.train_data_dir, "test_label.txt")
    
    if not os.path.exists(train_label):
        logger.warning(f"Prepared training data not found: {train_label}")
        logger.warning("You need to run 'convert_dataset.py' before training")
        warnings_found += 1
    
    if not os.path.exists(test_label):
        logger.warning(f"Prepared test data not found: {test_label}")
        logger.warning("You need to run 'convert_dataset.py' before training")
        warnings_found += 1
    
    # Summary
    if issues_found == 0 and warnings_found == 0:
        logger.info("✅ Detection dataset validation completed with no issues!")
        return True
    elif issues_found == 0:
        logger.info(f"⚠️ Detection dataset validation completed with {warnings_found} warnings.")
        return True
    else:
        logger.error(f"❌ Detection dataset validation completed with {issues_found} errors and {warnings_found} warnings.")
        return False

def validate_recognition_dataset():
    """Validate recognition dataset structure and format."""
    logger.info(f"Validating recognition dataset: {args.rec_dataset_dir}")
    
    issues_found = 0
    warnings_found = 0
    
    # Check directory structure
    if not os.path.exists(args.rec_dataset_dir):
        logger.error(f"Recognition dataset directory not found: {args.rec_dataset_dir}")
        return False
    
    images_dir = os.path.join(args.rec_dataset_dir, "images")
    if not os.path.exists(images_dir):
        logger.error(f"Images directory not found: {images_dir}")
        issues_found += 1
    
    # Check label files
    train_list = os.path.join(args.rec_dataset_dir, "train_list.txt")
    test_list = os.path.join(args.rec_dataset_dir, "test_list.txt")
    
    if not os.path.exists(train_list):
        logger.error(f"Training list file not found: {train_list}")
        issues_found += 1
    
    if not os.path.exists(test_list):
        logger.error(f"Test list file not found: {test_list}")
        issues_found += 1
    
    # Validate label format and image existence
    label_files = [('train', train_list), ('test', test_list)]
    for list_type, list_file in label_files:
        if not os.path.exists(list_file):
            continue
        
        logger.info(f"Validating {list_type} list: {list_file}")
        with open(list_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        logger.info(f"Found {len(lines)} entries in {list_type} list")
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                logger.warning(f"Empty line in {list_file} at line {i+1}")
                warnings_found += 1
                continue
                
            try:
                parts = line.split('\t')
                if len(parts) != 2:
                    logger.error(f"Invalid format in {list_file} at line {i+1}: expected 2 tab-separated parts, got {len(parts)}")
                    issues_found += 1
                    continue
                
                img_path, label = parts
                
                # Check if image exists
                full_img_path = os.path.join(args.rec_dataset_dir, "images", img_path)
                if not os.path.exists(full_img_path):
                    logger.error(f"Image not found: {full_img_path} (referenced in {list_file} line {i+1})")
                    issues_found += 1
                elif args.check_images:
                    valid, error = validate_image_file(full_img_path)
                    if not valid:
                        logger.error(f"Invalid image: {full_img_path} - {error}")
                        issues_found += 1
                
                # Check if label is empty or contains placeholder
                if not label:
                    logger.error(f"Empty label in {list_file} line {i+1}")
                    issues_found += 1
                elif label == "###":
                    logger.warning(f"Placeholder label '###' in {list_file} line {i+1} - you need to provide actual labels")
                    warnings_found += 1
                
            except Exception as e:
                logger.error(f"Error processing {list_file} line {i+1}: {str(e)}")
                issues_found += 1
    
    # Check character dictionary
    dict_path = "model_training/rec_train/meter_dict.txt"
    if not os.path.exists(dict_path):
        logger.warning(f"Character dictionary not found: {dict_path}")
        logger.warning("A default dictionary will be created during training")
        warnings_found += 1
    else:
        with open(dict_path, 'r') as f:
            chars = [line.strip() for line in f if line.strip()]
        logger.info(f"Found character dictionary with {len(chars)} characters")
    
    # Summary
    if issues_found == 0 and warnings_found == 0:
        logger.info("✅ Recognition dataset validation completed with no issues!")
        return True
    elif issues_found == 0:
        logger.info(f"⚠️ Recognition dataset validation completed with {warnings_found} warnings.")
        return True
    else:
        logger.error(f"❌ Recognition dataset validation completed with {issues_found} errors and {warnings_found} warnings.")
        return False

def main():
    """Main function to validate datasets."""
    try:
        logger.info("=" * 50)
        logger.info("PaddleOCR Dataset Validation")
        logger.info("=" * 50)
        
        # Setup environment
        setup_environment()
        
        # Validate detection dataset
        logger.info("\n" + "=" * 50)
        logger.info("DETECTION DATASET VALIDATION")
        logger.info("=" * 50)
        det_valid = validate_detection_dataset()
        
        # Validate recognition dataset
        logger.info("\n" + "=" * 50)
        logger.info("RECOGNITION DATASET VALIDATION")
        logger.info("=" * 50)
        rec_valid = validate_recognition_dataset()
        
        # Overall result
        logger.info("\n" + "=" * 50)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 50)
        
        if det_valid and rec_valid:
            logger.info("✅ All datasets are valid and ready for training!")
        else:
            if not det_valid:
                logger.error("❌ Detection dataset has issues that need to be addressed")
            if not rec_valid:
                logger.error("❌ Recognition dataset has issues that need to be addressed")
            logger.info("Please fix the issues above before proceeding with training")
    
    except Exception as e:
        logger.error(f"An error occurred during validation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 