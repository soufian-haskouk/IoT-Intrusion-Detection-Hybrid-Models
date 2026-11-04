# 🛡️ IoT Network Anomaly Detection using Hybrid Deep Learning (LSTM + Ensemble Classifiers)

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-orange.svg)](https://research.unsw.edu.au/unsw-nb15-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive comparative study and implementation of **Hybrid Deep Learning & Machine Learning architectures** designed to detect cyber attacks and network anomalies in **Internet of Things (IoT)** environments, evaluated on the benchmark **UNSW-NB15** dataset.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Architecture & Methodology](#-architecture--methodology)
- [Experimental Results](#-experimental-results)
- [Repository Structure](#-repository-structure)
- [Installation & Requirements](#-installation--requirements)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 🔍 Project Overview
Modern IoT infrastructures are increasingly vulnerable to sophisticated cyber threats. Due to resource constraints and complex traffic behaviors, traditional intrusion detection systems often fall short. 

This project explores a **hybrid modeling paradigm** that combines the temporal feature extraction power of **Deep Learning (LSTM)** with the robust decision-making of **Ensemble Classifiers (XGBoost & Random Forest)** to accurately classify malicious vs. benign network traffic.

### Proposed Hybrid Models:
1. **H1: LSTM + XGBoost** (Deep temporal extraction followed by Gradient Boosting classification)
2. **H2: LSTM + Random Forest** (Deep temporal extraction followed by Random Forest classification)

---

## ⚙️ Architecture & Methodology
* **Data Preprocessing & Feature Selection:** Utilizes `SelectKBest` to extract the top 25 most critical network traffic features from the UNSW-NB15 dataset.
* **Class Balancing:** Implements **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance between normal and attack traffic.
* **Feature Extraction:** Deep LSTM layers automatically learn sequential patterns, generating high-level latent feature representations (latent space dimensions: 32 for H1, 16 for H2).
* **Classification:** Final decision is delegated to optimized ensemble tree-based models (XGBoost / Random Forest).

---

## 📊 Experimental Results

A rigorous performance comparison between the two proposed architectures yielded the following metrics on the test set:

| Evaluation Metric | Model H1: LSTM + XGBoost | Model H2: LSTM + Random Forest |
| :--- | :---: | :---: |
| **Accuracy** | **90.45%** | 87.03% |
| **Weighted F1-Score** | **90.37%** | 86.90% |
| **Attack Recall (Detection Rate)** | **96.31%** | 93.73% |
| **ROC-AUC (Avg. Precision)** | **0.9754** | **0.9518** |
| **Training Execution Time** | 282.4 s | **217.8 s** |

### 🔑 Key Findings:
* **Model H1 (LSTM + XGBoost)** demonstrates superior overall accuracy (**90.45%**) and an exceptional attack detection capability (Recall of **96.31%**), making it the optimal solution for high-security IoT applications where minimizing false negatives is critical.
* **Model H2 (LSTM + Random Forest)** provides faster training execution (**217.8s**), serving as a reliable alternative for resource-constrained environments.

---
🚀 How to Run
Execute the training and evaluation pipeline for either model directly via Python:

Run Model H1 (LSTM + XGBoost):

Bash
python model_H1_LSTM_XGBoost.py
Run Model H2 (LSTM + Random Forest):

Bash
python model_H2_LSTM_RF.py
👨‍💻 Author
Soufian Haskouk
## 📁 Repository Structure
```text
IoT-Intrusion-Detection-Hybrid-Models/
│
├── DATA/                   # Dataset directory (UNSW-NB15 subset)
├── models/                 # Saved model weights (.h5, .joblib, and compressed archives)
├── imageH1/                # Evaluation plots, confusion matrices, and ROC curves for H1
├── imageH2/                # Evaluation plots, confusion matrices, and ROC curves for H2
├── model_H1_LSTM_XGBoost.py# Main Python script for Model H1 training & evaluation
├── model_H2_LSTM_RF.py     # Main Python script for Model H2 training & evaluation
├── requirements.txt        # Required Python libraries and dependencies
└── README.md               # Project documentation
