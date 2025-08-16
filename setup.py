"""
Setup script for TabularBERT package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt if it exists
def read_requirements():
    requirements_path = "requirements.txt"
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return []

setup(
    name="tabularbert",
    version="0.1.0",
    author="Beomjin Park",
    author_email="bbeomjin@gmail.com",
    description="A comprehensive framework for tabular data modeling using BERT-based transformers",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/bbeomjin/TabularBERT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "tensorboard>=2.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "wandb": ["wandb>=0.12.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.9",
        ],
        "all": ["wandb>=0.12.0"],
    },
    include_package_data=True,
    package_data={
        "tabularbert": ["*.md", "*.txt"],
    },
    keywords="tabular-data, bert, transformer, machine-learning, deep-learning, self-supervised",
    project_urls={
        "Bug Reports": "https://github.com/bbeomjin/tabularbert/issues",
        "Source": "https://github.com/bbeomjin/tabularbert",
        "Documentation": "https://tabularbert.readthedocs.io/",
    },
)
