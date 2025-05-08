#!/usr/bin/env python
# coding: utf-8

"""
Script to push GitHub repository to Kaggle as a dataset
This allows easy integration between GitHub and Kaggle for training
"""

import os
import sys
import subprocess
import argparse
import json
import tempfile
import shutil
from pathlib import Path

def setup_kaggle_api():
    """Check and setup Kaggle API credentials"""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print("Kaggle API credentials not found.")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/settings and create an API token")
        print("2. Download the kaggle.json file")
        print("3. Place it in ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Check if kaggle command is available
    try:
        subprocess.run(["kaggle", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("Kaggle CLI not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"])
    
    return True

def create_dataset_from_github(github_repo, dataset_name, dataset_title, description):
    """Create a Kaggle dataset from a GitHub repository"""
    # Clone repository to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Cloning repository {github_repo} to temporary directory...")
        clone_cmd = f"git clone https://github.com/{github_repo}.git {temp_dir}"
        result = subprocess.run(clone_cmd, shell=True, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print(f"Error cloning repository: {result.stderr.decode()}")
            return False
        
        # Remove .git directory to avoid issues
        git_dir = os.path.join(temp_dir, ".git")
        if os.path.exists(git_dir):
            shutil.rmtree(git_dir)
        
        # Create dataset metadata
        dataset_dir = Path(temp_dir)
        metadata = {
            "title": dataset_title,
            "id": f"{os.environ.get('KAGGLE_USERNAME', 'username')}/{dataset_name}",
            "licenses": [{"name": "MIT"}],
            "description": description
        }
        
        with open(dataset_dir / "dataset-metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Push to Kaggle
        print(f"Pushing to Kaggle as dataset {dataset_name}...")
        kaggle_cmd = f"kaggle datasets create -p {temp_dir} -r zip"
        result = subprocess.run(kaggle_cmd, shell=True)
        
        if result.returncode != 0:
            print("Error creating Kaggle dataset")
            return False
        
        print(f"Successfully created Kaggle dataset: {dataset_name}")
        print(f"You can now reference this dataset in your Kaggle notebooks")
        return True

def create_dataset_from_data(data_dir, dataset_name, dataset_title, description):
    """Create a Kaggle dataset from a local data directory"""
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist")
        return False
    
    # Create dataset metadata
    dataset_dir = Path(data_dir)
    metadata = {
        "title": dataset_title,
        "id": f"{os.environ.get('KAGGLE_USERNAME', 'username')}/{dataset_name}",
        "licenses": [{"name": "MIT"}],
        "description": description
    }
    
    metadata_file = dataset_dir / "dataset-metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Push to Kaggle
    print(f"Pushing to Kaggle as dataset {dataset_name}...")
    kaggle_cmd = f"kaggle datasets create -p {data_dir} -r zip"
    result = subprocess.run(kaggle_cmd, shell=True)
    
    if result.returncode != 0:
        print("Error creating Kaggle dataset")
        return False
    
    print(f"Successfully created Kaggle dataset: {dataset_name}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Push GitHub repository or data directory to Kaggle as a dataset")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # GitHub repository to Kaggle dataset
    github_parser = subparsers.add_parser("github", help="Create dataset from GitHub repository")
    github_parser.add_argument("--repo", required=True, help="GitHub repository (username/repo)")
    github_parser.add_argument("--name", required=True, help="Kaggle dataset name")
    github_parser.add_argument("--title", help="Kaggle dataset title")
    github_parser.add_argument("--description", default="Dataset created from GitHub repository", 
                             help="Dataset description")
    
    # Local data directory to Kaggle dataset
    data_parser = subparsers.add_parser("data", help="Create dataset from local data directory")
    data_parser.add_argument("--dir", required=True, help="Data directory path")
    data_parser.add_argument("--name", required=True, help="Kaggle dataset name")
    data_parser.add_argument("--title", help="Kaggle dataset title")
    data_parser.add_argument("--description", default="Training data for PaddleOCR",
                           help="Dataset description")
    
    args = parser.parse_args()
    
    if not setup_kaggle_api():
        return
    
    if args.command == "github":
        title = args.title or f"GitHub Repository: {args.repo}"
        create_dataset_from_github(args.repo, args.name, title, args.description)
    elif args.command == "data":
        title = args.title or f"Dataset: {args.name}"
        create_dataset_from_data(args.dir, args.name, title, args.description)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 