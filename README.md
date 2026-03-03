# Melanoma Detection using Computer Vision

🚧 **Project Status:** Initial setup and architectural planning phase.

This repository contains a team-based, end-to-end machine learning system exploring melanoma detection from dermoscopic images. The project emphasises reproducibility, structured evaluation, responsible deployment, and collaborative engineering practices.

---

## Project Overview

Melanoma is one of the most serious forms of skin cancer, where early detection significantly improves outcomes. This project explores the use of computer vision techniques to classify dermoscopic images as melanoma or non-melanoma.

The goal is to demonstrate a complete ML workflow:
- Dataset handling
- Model training
- Evaluation and analysis
- Mobile deployment
- Team-based software engineering discipline

---

## Scope (High-Level)

- Binary classification: melanoma vs non-melanoma
- Baseline CNN model trained from scratch
- Evaluation prioritising sensitivity (recall for melanoma class)
- Structured performance reporting (confusion matrix, ROC curve, metric table)
- Android demo application with on-device inference

---

## Repository Structure (Planned)
Melanoma-CV-Project/
├── training/ # Python training & evaluation pipeline
├── android/ # Android application (Java)
├── docs/ # Architecture and technical documentation
├── PROJECT_SCOPE.md # Internal scope definition
├── DECISIONS.md # Technical decision log
├── CONTRIBUTING.md # Collaboration rules
└── README.md

## Engineering Principles

- Reproducibility over performance hype
- Explicit handling of class imbalance
- Patient-level data leakage prevention
- Clear separation of training and deployment code
- Structured pull request workflow

---

## Disclaimer

This project is strictly for educational and portfolio purposes.  
It is not intended for clinical or diagnostic use.  
Results must not be relied upon for medical decision-making.

## Setup Instructions

#Create a virtual environment
python -m venv .venv

#Activate environment
Windows:
.venv\Scripts\activate

Mac/Linux:
source .venv/bin/activate

#Install dependencies
pip install -r requirements.txt

#Run program
python main.py
