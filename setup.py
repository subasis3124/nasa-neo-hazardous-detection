from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements() -> List[str]:
    """
    Reads requirements.txt and returns a list of dependencies.
    Filters out '-e .' which is used for editable installs.
    """
    requirement_list: List[str] = []
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as file:
            for line in file:
                requirement = line.strip()
                if requirement and requirement != HYPHEN_E_DOT:
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("requirements.txt not found!!!")
    return requirement_list

setup(
    name="nasa-neo-hazardous-detection",
    version="0.0.1",
    author="Subasis Mishra",
    author_email="subasismishra3124@gmail.com",
    description=(
        "A production-grade machine learning pipeline for NASA Near-Earth Object (NEO) hazard classification. "
        "The project automates weekly data ingestion from NASA, data preprocessing with SMOTE, "
        "model training with MLflow tracking on DAGsHub, and logs the best model and preprocessing pipeline "
        "into PostgreSQL. It also includes automated data drift detection, Grafana dashboards for monitoring, "
        "and email/Grafana alerts for anomalies."
    ),
    url="https://github.com/Subrat1920/Nasa-Near-Earth-Hazardous-Detection",
    packages=find_packages(),
    install_requires=get_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database :: Database Engines/Servers",
    ],
    python_requires=">=3.11",
)
