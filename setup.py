"""Setup script for NEPSE Price Prediction System"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nepse-price-prediction",
    version="0.1.0",
    author="Hamro Polyclinic",
    author_email="hamropolyclinic@gmail.com",
    description="An end-to-end machine learning system for predicting NEPSE stock prices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hamropolyclinic/nepse-price-prediction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nepse-collector=data_collection.collector:main",
            "nepse-train=train:main",
        ],
    },
)
