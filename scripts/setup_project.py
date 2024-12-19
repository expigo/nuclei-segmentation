# scripts/setup_project.py
import os
import subprocess
import argparse

def setup_project(machine_type):
    """Setup project for specific machine."""
    # Install dependencies
    subprocess.run(["mamba", "env", "create", "-f", "environment.yml"])
    
    # Setup DVC
    if not os.path.exists(".dvc"):
        subprocess.run(["dvc", "init"])
        subprocess.run(["dvc", "remote", "add", "-d", "myremote", 
                       "gdrive://your-drive-folder-id"])
    
    # Pull data
    subprocess.run(["dvc", "pull"])
    
    # Login to wandb
    subprocess.run(["wandb", "login"])
    
    print(f"Project setup completed for {machine_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, required=True,
                       choices=["macbook_m3", "ubuntu_rtx3080"])
    args = parser.parse_args()
    setup_project(args.machine)