from setuptools import setup, find_packages

setup(
    name="nuclei_segmentation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'scipy',
        'scikit-image',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'wandb',
        'opencv-python',
        'albumentations',
        'dvc',
        'pytorch-lightning',
        'hydra-core',
    ]
)