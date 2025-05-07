import os
import json
import argparse


def convert_to_paddleocr_format(input_file, output_file):
    """
    Convert the dataset format to PaddleOCR's text detection format.
    PaddleOCR expects tab-separated format:
    image_path\tpoint1_x,point1_y,point2_x,point2_y,point3_x,point3_y,point4_x,point4_y\ttranscription
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Warning: Unexpected format in line: {line}")
                continue
                
            img_path, label_info = parts
            
            try:
                annotations = json.loads(label_info)
                for ann in annotations:
                    transcription = ann.get("transcription", "###")
                    points = ann.get("points", [])
                    
                    if len(points) == 4:  # Make sure we have a quadrilateral
                        # Flatten the points list to the format: x1,y1,x2,y2,x3,y3,x4,y4
                        coords = ','.join([str(int(coord)) for point in points for coord in point])
                        
                        # Write in paddleocr format
                        f_out.write(f"{img_path}\t{coords}\t{transcription}\n")
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in line for {img_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert dataset to PaddleOCR format')
    parser.add_argument('--dataset_dir', type=str, default='dataset/det_dataset_1',
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='train_data/meter_detection',
                        help='Output directory for converted dataset')
    args = parser.parse_args()
    
    # Convert train dataset
    train_input = os.path.join(args.dataset_dir, "train_verified.txt")
    train_output = os.path.join(args.output_dir, "train_label.txt")
    
    # Convert test dataset
    test_input = os.path.join(args.dataset_dir, "test_verified.txt")
    test_output = os.path.join(args.output_dir, "test_label.txt")
    
    print(f"Converting training data: {train_input} -> {train_output}")
    convert_to_paddleocr_format(train_input, train_output)
    
    print(f"Converting test data: {test_input} -> {test_output}")
    convert_to_paddleocr_format(test_input, test_output)
    
    print(f"Done! Converted files saved to {args.output_dir}")
    print("Note: You'll need to link or copy your images to the train_data directory")
    print("Suggested command:")
    print(f"mkdir -p {args.output_dir}/images")
    print(f"ln -s $(pwd)/{args.dataset_dir}/images/* {args.output_dir}/images/")


if __name__ == "__main__":
    main() 