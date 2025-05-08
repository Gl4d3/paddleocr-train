#!/usr/bin/env python
# coding: utf-8

"""
Setup script for local MLflow tracking server
"""

import os
import argparse
import subprocess
import socket
import time
import sys

def check_port(host, port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def start_mlflow_server(host='localhost', port=5000, backend_uri=None, artifact_uri=None):
    """Start MLflow tracking server"""
    # Check if port is already in use
    if check_port(host, port):
        print(f"Port {port} is already in use. MLflow server may already be running.")
        return False
    
    # Prepare command
    cmd = f"mlflow server --host {host} --port {port}"
    
    # Add backend store URI if provided
    if backend_uri:
        cmd += f" --backend-store-uri {backend_uri}"
    else:
        # Default to a local SQLite database
        os.makedirs("mlruns_db", exist_ok=True)
        cmd += " --backend-store-uri sqlite:///mlruns_db/mlflow.db"
    
    # Add artifact store URI if provided
    if artifact_uri:
        cmd += f" --default-artifact-root {artifact_uri}"
    
    # Print command
    print(f"Starting MLflow server with command: {cmd}")
    
    # Start server
    try:
        # Start server in a separate process
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit to check if server started properly
        time.sleep(2)
        if process.poll() is not None:
            # Process has terminated
            stdout, stderr = process.communicate()
            print("Error starting MLflow server:")
            print(stderr.decode())
            return False
        
        print(f"MLflow server started on http://{host}:{port}")
        print("To stop the server, press Ctrl+C")
        
        # Write connection info to file for other scripts to use
        with open("mlflow_connection.txt", "w") as f:
            f.write(f"http://{host}:{port}")
        
        # Wait for server to run
        try:
            process.wait()
        except KeyboardInterrupt:
            print("Stopping MLflow server...")
            process.terminate()
        
        return True
    
    except Exception as e:
        print(f"Error starting MLflow server: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Start a local MLflow tracking server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--backend-store-uri", help="URI for backend store (default: SQLite)")
    parser.add_argument("--default-artifact-root", help="Directory for storing artifacts")
    
    args = parser.parse_args()
    
    # Check if MLflow is installed
    try:
        import mlflow
    except ImportError:
        print("MLflow not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "mlflow"])
    
    # Start server
    start_mlflow_server(
        host=args.host,
        port=args.port,
        backend_uri=args.backend_store_uri,
        artifact_uri=args.default_artifact_root
    )

if __name__ == "__main__":
    main() 