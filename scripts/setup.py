import sys
import os
import pkgutil
import subprocess
import bpy


dir_scripts = os.path.dirname(bpy.context.space_data.text.filepath)
dir_packages = os.path.join(os.path.dirname(dir_scripts), 'packages')
if not os.path.isdir(dir_packages):
    print('Creating package directory...')
    os.makedirs(dir_packages)

# Get the path to the Python interpreter used by Blender
python_path = sys.executable
subprocess.run([python_path, "-m", "ensurepip"])
# Ensure pip is up to date
subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"])

# Check if the required packages are installed
required_packages = "numpy", "scipy", "pyyaml", "einops", "timm==0.9.2", "trimesh"]
installed_packages = {pkg.name for pkg in pkgutil.iter_modules()}

for package in required_packages:
    if package not in installed_packages:
        # Install the required package in the created folder
        subprocess.run([python_path, "-m", "pip", "install", "--target=" + dir_packages, package])
    else:
        print(f"{package} is already installed")

# Install pytorch
if "torch" not in installed_packages:
    subprocess.run([python_path, "-m", "pip", "install", "torch", "torchvision", "--target=" + dir_packages,
                    "--index-url=https://download.pytorch.org/whl/cu118"])

print('Done.')

print('Testing...')
sys.path.append(dir_packages)
import trimesh
import numpy as np
import torch
print(torch.cuda.is_available())
print('Testing successful. Done.')
