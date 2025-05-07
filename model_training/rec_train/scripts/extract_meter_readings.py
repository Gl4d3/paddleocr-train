import os
import cv2
import numpy as np
import json
import random
import argparse
from tqdm import tqdm

def extract_text_regions(image_path, annotation_file, output_dir, margin=5):
    """
    Extract text regions from images based on detection annotations.
    Adds a small margin around the detected region to ensure the text is fully captured.
    """
    # Read annotations
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()
    
    # Process each annotation
    extracted_count = 0
    for annotation in tqdm(annotations, desc="Extracting text regions"):
        parts = annotation.strip().split('\t')
        if len(parts) < 2:
            continue
            
        img_path, coords = parts[0], parts[1]
        if len(parts) >= 3:
            text_label = parts[2]
        else:
            text_label = "###"  # Unknown text label
            
        # Convert coordinates to integers
        try:
            points = list(map(int, coords.split(',')))
            if len(points) != 8:  # Should have 4 points with x,y coordinates
                continue
                
            # Reshape points into 4 (x,y) coordinates
            points = np.array(points).reshape(4, 2)
            
            # Read image
            full_img_path = os.path.join(image_path, img_path)
            if not os.path.exists(full_img_path):
                print(f"Warning: Image {full_img_path} not found")
                continue
                
            img = cv2.imread(full_img_path)
            if img is None:
                print(f"Warning: Could not read image {full_img_path}")
                continue
                
            # Get bounding rectangle with margin
            x_min = max(0, min(p[0] for p in points) - margin)
            y_min = max(0, min(p[1] for p in points) - margin)
            x_max = min(img.shape[1], max(p[0] for p in points) + margin)
            y_max = min(img.shape[0], max(p[1] for p in points) + margin)
            
            # Extract region
            roi = img[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                print(f"Warning: Empty ROI in {img_path}")
                continue
                
            # Save image
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_img_path = os.path.join(output_dir, f"{base_name}_roi_{extracted_count}.jpg")
            cv2.imwrite(output_img_path, roi)
            
            # Add entry to labels
            yield output_img_path.replace(output_dir + '/', ''), text_label
            extracted_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Extracted {extracted_count} text regions")


def create_train_val_lists(data_dir, output_dir, train_ratio=0.8):
    """
    Create train and validation lists from the extracted text regions.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    random.shuffle(all_files)
    
    # Split into train and validation sets
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Total images: {len(all_files)}")
    print(f"Train images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    
    # Create train and validation lists with dummy labels
    # (To be updated with actual OCR labels later)
    with open(os.path.join(output_dir, 'train_list.txt'), 'w') as f:
        for img_file in train_files:
            f.write(f"{img_file}\t###\n")
    
    with open(os.path.join(output_dir, 'val_list.txt'), 'w') as f:
        for img_file in val_files:
            f.write(f"{img_file}\t###\n")


def main():
    parser = argparse.ArgumentParser(description='Extract text regions for recognition training')
    parser.add_argument('--det_data_dir', type=str, default='dataset/det_dataset_1',
                       help='Path to detection dataset directory')
    parser.add_argument('--rec_output_dir', type=str, default='dataset/rec_dataset_1',
                       help='Output directory for recognition dataset')
    parser.add_argument('--train_data', type=str, default='train_data/meter_detection/train_label.txt',
                       help='Path to detection training data labels')
    parser.add_argument('--test_data', type=str, default='train_data/meter_detection/test_label.txt',
                       help='Path to detection test data labels')
    parser.add_argument('--margin', type=int, default=5,
                       help='Margin to add around detected regions')
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.rec_output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.rec_output_dir, 'images'), exist_ok=True)
    
    # Extract text regions from training and test data
    print("Processing training data...")
    train_labels = list(extract_text_regions(
        args.det_data_dir,
        args.train_data,
        os.path.join(args.rec_output_dir, 'images'),
        args.margin
    ))
    
    print("Processing test data...")
    test_labels = list(extract_text_regions(
        args.det_data_dir,
        args.test_data,
        os.path.join(args.rec_output_dir, 'images'),
        args.margin
    ))
    
    # Write the labels to files
    with open(os.path.join(args.rec_output_dir, 'train_list.txt'), 'w') as f:
        for img_path, label in train_labels:
            f.write(f"{img_path}\t{label}\n")
    
    with open(os.path.join(args.rec_output_dir, 'test_list.txt'), 'w') as f:
        for img_path, label in test_labels:
            f.write(f"{img_path}\t{label}\n")
    
    print(f"Created recognition dataset at {args.rec_output_dir}")
    print(f"Training samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print("\nNOTE: The labels are currently set to '###'. You will need to manually annotate the actual text for recognition training.")
    
if __name__ == "__main__":
    main() 