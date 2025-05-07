import os
import json
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon


def read_label_file(label_file):
    """Read label file and return a dictionary of image path to boxes"""
    data = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                print(f"Warning: Unexpected format in line: {line}")
                continue
                
            img_path, label_info = parts
            try:
                annotations = json.loads(label_info)
                boxes = []
                for ann in annotations:
                    points = ann.get("points", [])
                    if len(points) == 4:  # Make sure it's a quadrilateral
                        boxes.append(np.array(points).astype(np.int32))
                data[img_path] = boxes
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in line for {img_path}")
    
    return data


def visualize_samples(dataset_path, label_file, output_file, num_samples=5):
    """Visualize random samples from the dataset with bounding boxes"""
    # Read the label file
    label_path = os.path.join(dataset_path, label_file)
    data = read_label_file(label_path)
    
    # Select random samples
    image_paths = list(data.keys())
    if len(image_paths) < num_samples:
        num_samples = len(image_paths)
    
    selected_paths = random.sample(image_paths, num_samples)
    
    # Create a figure to display the images
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, num_samples * 5))
    if num_samples == 1:
        axes = [axes]
    
    # Display each image with bounding boxes
    for i, img_path in enumerate(selected_paths):
        full_img_path = os.path.join(dataset_path, img_path)
        
        # Read the image
        img = cv2.imread(full_img_path)
        if img is None:
            print(f"Warning: Could not read image {full_img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display the image
        axes[i].imshow(img)
        axes[i].set_title(f"Image: {os.path.basename(img_path)}")
        
        # Draw bounding boxes
        for box in data[img_path]:
            # Convert to appropriate format for Polygon
            box_array = np.array(box)
            polygon = Polygon(box_array, fill=False, edgecolor='red', linewidth=2)
            axes[i].add_patch(polygon)
            
            # Add text to indicate the corner points
            for j, (x, y) in enumerate(box):
                axes[i].text(x, y, f"{j}", color='white', fontsize=12, 
                           bbox=dict(facecolor='red', alpha=0.7))
        
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Visualization saved to {output_file}")


def main():
    dataset_path = "dataset/det_dataset_1"
    
    # Test on training set
    print("Visualizing samples from training set...")
    visualize_samples(dataset_path, "train_verified.txt", "model_training/det_train/visualizations/train_visualization.png", num_samples=3)
    
    # Test on test set
    print("Visualizing samples from test set...")
    visualize_samples(dataset_path, "test_verified.txt", "model_training/det_train/visualizations/test_visualization.png", num_samples=3)


if __name__ == "__main__":
    main() 