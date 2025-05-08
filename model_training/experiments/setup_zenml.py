#!/usr/bin/env python
# coding: utf-8

"""
Setup script for ZenML with MLflow integration
This script helps set up ZenML with an MLflow tracking stack
"""

import os
import sys
import subprocess
import argparse

def check_installation():
    """Check if ZenML and MLflow are installed"""
    try:
        import zenml
        print(f"ZenML version: {zenml.__version__}")
    except ImportError:
        print("ZenML not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "zenml"])
    
    try:
        import mlflow
        print(f"MLflow version: {mlflow.__version__}")
    except ImportError:
        print("MLflow not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "mlflow"])

def setup_zenml(mlflow_tracking_uri=None):
    """Set up ZenML with MLflow tracking"""
    from zenml.client import Client
    
    # Initialize ZenML if not already initialized
    try:
        client = Client()
    except:
        print("Initializing ZenML...")
        subprocess.run(["zenml", "init"])
        client = Client()
    
    # Register MLflow tracking component
    try:
        # Check if MLflow component exists
        try:
            client.get_stack_component("mlflow_tracker")
            print("MLflow tracking component already exists.")
        except:
            print("Registering MLflow tracking component...")
            if mlflow_tracking_uri:
                # Use provided tracking URI
                subprocess.run([
                    "zenml", "experiment-tracker", "register", "mlflow_tracker",
                    "--flavor", "mlflow",
                    "--tracking_uri", mlflow_tracking_uri
                ])
            else:
                # Use local file-based tracking
                os.makedirs("mlruns", exist_ok=True)
                tracking_uri = f"file://{os.path.abspath('mlruns')}"
                subprocess.run([
                    "zenml", "experiment-tracker", "register", "mlflow_tracker",
                    "--flavor", "mlflow",
                    "--tracking_uri", tracking_uri
                ])
                print(f"Registered MLflow tracker with local tracking URI: {tracking_uri}")
        
        # Register stack
        try:
            # Check if stack exists
            client.get_stack("mlflow_stack")
            print("Stack 'mlflow_stack' already exists.")
        except:
            print("Registering new stack with MLflow...")
            subprocess.run([
                "zenml", "stack", "register", "mlflow_stack",
                "-e", "mlflow_tracker"
            ])
        
        # Activate the stack
        subprocess.run(["zenml", "stack", "set", "mlflow_stack"])
        print("Stack 'mlflow_stack' is now active.")
        
        return True
    except Exception as e:
        print(f"Error setting up ZenML with MLflow: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Set up ZenML with MLflow integration")
    parser.add_argument("--tracking-uri", help="MLflow tracking URI (default: local file-based tracking)")
    
    args = parser.parse_args()
    
    # Check installations
    check_installation()
    
    # Set up ZenML
    setup_zenml(args.tracking_uri)
    
    print("Setup complete. You can now run ZenML pipelines with MLflow tracking.")
    print("To start an MLflow UI, run: mlflow ui")

if __name__ == "__main__":
    main() 