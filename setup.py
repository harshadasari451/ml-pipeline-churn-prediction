from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ml-pipeline-churn-prediction",
    version="1.0.0",
    author="ML Pipeline Team",
    author_email="",
    description="End-to-End ML Pipeline for Customer Churn Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harshadasari451/ml-pipeline-churn-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ml-pipeline=src.main:main",
        ],
    },
)
