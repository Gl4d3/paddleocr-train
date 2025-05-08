#!/usr/bin/env python
# coding: utf-8

"""
Script to trigger PaddleOCR training on Kaggle via their API
"""

import os
import sys
import argparse
import time
import json
from kaggle.api.kaggle_api_extended import KaggleApi

def setup_kaggle_api():
    """Set up and authenticate Kaggle API"""
    api = KaggleApi()
    try:
        api.authenticate()
        print("Kaggle API authenticated successfully")
        return api
    except Exception as e:
        print(f"Error authenticating Kaggle API: {str(e)}")
        print("Make sure you have kaggle.json in ~/.kaggle/ with the correct permissions")
        sys.exit(1)

def create_kernel_metadata(notebook_path, output_path, dataset_slug=None, gpu_enabled=True):
    """Create metadata for the kernel"""
    kernel_metadata = {
        "id": f"{os.environ.get('KAGGLE_USERNAME', 'username')}/paddleocr-training",
        "title": "PaddleOCR Training Pipeline",
        "code_file": os.path.basename(notebook_path),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu_enabled,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }
    
    # Add dataset if specified
    if dataset_slug:
        kernel_metadata["dataset_sources"] = [dataset_slug]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Write metadata to file
    with open(os.path.join(output_path, "kernel-metadata.json"), "w") as f:
        json.dump(kernel_metadata, f, indent=2)
    
    # Copy notebook to output directory
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook file {notebook_path} not found")
        sys.exit(1)
    
    import shutil
    shutil.copy2(notebook_path, os.path.join(output_path, os.path.basename(notebook_path)))
    
    return os.path.join(output_path, "kernel-metadata.json")

def push_and_run_kernel(api, kernel_path):
    """Push and run the kernel"""
    try:
        print(f"Pushing kernel from {kernel_path}")
        api.kernels_push(kernel_path)
        
        # Extract kernel name from metadata
        with open(os.path.join(kernel_path, "kernel-metadata.json"), "r") as f:
            metadata = json.load(f)
            kernel_slug = metadata["id"]
        
        print(f"Kernel pushed successfully with slug: {kernel_slug}")
        print("Starting kernel run...")
        api.kernels_status_cli(kernel_slug)
        
        return kernel_slug
    except Exception as e:
        print(f"Error pushing or running kernel: {str(e)}")
        return None

def wait_for_completion(api, kernel_slug, check_interval=60, timeout=None):
    """Wait for the kernel to complete"""
    start_time = time.time()
    print("Waiting for kernel to complete...")
    
    while True:
        # Check if timeout has been exceeded
        if timeout and (time.time() - start_time > timeout):
            print(f"Timeout of {timeout} seconds exceeded")
            return False
        
        # Get kernel status
        kernel = api.kernel_status(kernel_slug)
        status = kernel["status"]
        
        if status == "complete":
            print("Kernel completed successfully!")
            return True
        elif status == "failed":
            print("Kernel failed!")
            return False
        elif status == "canceled":
            print("Kernel was canceled.")
            return False
        else:
            print(f"Kernel status: {status}, waiting {check_interval} seconds...")
            time.sleep(check_interval)

def download_results(api, kernel_slug, output_path):
    """Download the results from the kernel"""
    try:
        print(f"Downloading results to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        api.kernels_output(kernel_slug, path=output_path)
        print("Results downloaded successfully")
        return True
    except Exception as e:
        print(f"Error downloading results: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Trigger PaddleOCR training on Kaggle')
    parser.add_argument('--notebook', required=True, help='Path to the notebook file to run')
    parser.add_argument('--output', default='./kaggle_run', help='Output directory for kernel files and results')
    parser.add_argument('--dataset', help='Kaggle dataset slug to use (e.g. "username/dataset-name")')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU for the kernel')
    parser.add_argument('--wait', action='store_true', help='Wait for kernel to complete')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds for kernel completion')
    parser.add_argument('--download', action='store_true', help='Download results after completion')
    
    args = parser.parse_args()
    
    # Setup Kaggle API
    api = setup_kaggle_api()
    
    # Create kernel metadata
    kernel_path = os.path.dirname(create_kernel_metadata(args.notebook, args.output, args.dataset, args.gpu))
    
    # Push and run kernel
    kernel_slug = push_and_run_kernel(api, kernel_path)
    if not kernel_slug:
        return
    
    # Wait for completion if requested
    if args.wait:
        completed = wait_for_completion(api, kernel_slug, timeout=args.timeout)
        if completed and args.download:
            download_results(api, kernel_slug, os.path.join(args.output, "results"))
    else:
        print(f"Kernel is running with slug: {kernel_slug}")
        print("To check status: kaggle kernels status " + kernel_slug)
        print("To download results: kaggle kernels output " + kernel_slug)

if __name__ == "__main__":
    main() 