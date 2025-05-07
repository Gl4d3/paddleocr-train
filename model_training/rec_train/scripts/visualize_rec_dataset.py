import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse

def visualize_recognition_dataset(data_dir, label_file, output_file, num_samples=10):
    """
    Visualize samples from the recognition dataset with their labels
    """
    # Read the label file
    with open(os.path.join(data_dir, label_file), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Select random samples
    if len(lines) > num_samples:
        samples = random.sample(lines, num_samples)
    else:
        samples = lines
    
    # Create a figure to display the images
    fig, axes = plt.subplots(len(samples), 1, figsize=(12, 2 * len(samples)))
    if len(samples) == 1:
        axes = [axes]
    
    # Display each image with its label
    for i, line in enumerate(samples):
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
            
        img_path, text_label = parts[0], parts[1]
        
        # Read the image
        img = cv2.imread(os.path.join(data_dir, img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display the image
        axes[i].imshow(img)
        axes[i].set_title(f"Label: '{text_label}' - Path: {os.path.basename(img_path)}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Visualization saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize recognition dataset')
    parser.add_argument('--data_dir', type=str, default='dataset/rec_dataset_1',
                        help='Path to the recognition dataset directory')
    parser.add_argument('--train_label', type=str, default='train_list.txt',
                        help='Name of the training label file')
    parser.add_argument('--test_label', type=str, default='test_list.txt',
                        help='Name of the test label file')
    parser.add_argument('--output_dir', type=str, default='model_training/rec_train/visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize training samples
    print("Visualizing training samples...")
    train_output = os.path.join(args.output_dir, 'train_visualization.png')
    visualize_recognition_dataset(args.data_dir, args.train_label, train_output, args.num_samples)
    
    # Visualize test samples
    print("Visualizing test samples...")
    test_output = os.path.join(args.output_dir, 'test_visualization.png')
    visualize_recognition_dataset(args.data_dir, args.test_label, test_output, args.num_samples)


if __name__ == "__main__":
    main() 