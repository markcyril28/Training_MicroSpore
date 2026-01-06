"""
Microspore Training - Setup Template
This template is used when modules/setup.py doesn't exist.
"""
from setuptools import setup, find_packages

setup(
    name="microspore_training",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.8",
    author="Microspore Phenotyping Team",
    description="Training modules for microspore phenotyping YOLO models",
)
