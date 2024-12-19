# scripts/verify_dvc.py
from pathlib import Path
import subprocess

def test_dvc_setup():
    print("Testing DVC setup...")
    
    # Check if DVC is initialized
    dvc_dir = Path('.dvc')
    if not dvc_dir.exists():
        print("Initializing DVC...")
        subprocess.run(['dvc', 'init'])
        subprocess.run(['git', 'add', '.dvc'])
        subprocess.run(['git', 'commit', '-m', "Initialize DVC"])
    else:
        print("DVC already initialized")
    
    # Test DVC commands
    try:
        subprocess.run(['dvc', 'status'], check=True)
        print("DVC status check successful")
    except subprocess.CalledProcessError as e:
        print(f"DVC status check failed: {str(e)}")

if __name__ == "__main__":
    test_dvc_setup()