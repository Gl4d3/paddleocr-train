#!/usr/bin/env python3
"""
Script to convert Python scripts to Jupyter notebooks.
This makes it easy to create and maintain notebooks in version control.
"""

import os
import sys
import argparse
import subprocess

def convert_to_notebook(python_file, output_dir=None):
    """Convert a Python file to a Jupyter notebook"""
    if not os.path.exists(python_file):
        print(f"Error: File {python_file} not found.")
        return False
    
    # Check if jupytext is installed
    try:
        import jupytext
    except ImportError:
        print("Jupytext not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "jupytext"], check=True)
        import jupytext
    
    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(python_file)
    
    base_name = os.path.basename(python_file)
    name_without_ext = os.path.splitext(base_name)[0]
    notebook_file = os.path.join(output_dir, f"{name_without_ext}.ipynb")
    
    print(f"Converting {python_file} to {notebook_file}...")
    
    # Convert Python file to notebook
    notebook = jupytext.read(python_file)
    jupytext.write(notebook, notebook_file)
    
    print(f"Successfully created {notebook_file}")
    return True

def convert_all_scripts(directory, output_dir=None):
    """Convert all Python scripts in a directory to notebooks"""
    if not os.path.isdir(directory):
        print(f"Error: Directory {directory} not found.")
        return
    
    if output_dir is None:
        output_dir = directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all Python files in the directory, but exclude this script
    script_name = os.path.basename(__file__)
    python_files = [f for f in os.listdir(directory) 
                   if f.endswith('.py') and f != script_name]
    
    if not python_files:
        print(f"No Python files found in {directory}.")
        return
    
    print(f"Found {len(python_files)} Python files to convert:")
    for py_file in python_files:
        print(f"  - {py_file}")
    
    # Convert each file
    for py_file in python_files:
        full_path = os.path.join(directory, py_file)
        convert_to_notebook(full_path, output_dir)

def main():
    parser = argparse.ArgumentParser(description="Convert Python scripts to Jupyter notebooks")
    parser.add_argument('--file', '-f', help="Specific Python file to convert")
    parser.add_argument('--dir', '-d', default=os.path.dirname(__file__), 
                        help="Directory containing Python files to convert")
    parser.add_argument('--output', '-o', help="Output directory for notebooks")
    
    args = parser.parse_args()
    
    # Install required packages
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "jupytext", "notebook"], check=True)
    except subprocess.CalledProcessError:
        print("Warning: Failed to install required packages.")
    
    if args.file:
        convert_to_notebook(args.file, args.output)
    else:
        convert_all_scripts(args.dir, args.output)

if __name__ == "__main__":
    main() 