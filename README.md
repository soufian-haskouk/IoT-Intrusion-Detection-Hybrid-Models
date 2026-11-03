# IoT Network Anomaly Detection using Hybrid Deep Learning (LSTM + Ensemble Classifiers)

This repository contains a comparative study and implementation of hybrid deep learning models designed to detect cyber attacks and anomalies in Internet of Things (IoT) environments using the **UNSW-NB15** dataset.

---

## 🚀 Overview
IoT networks are increasingly vulnerable to sophisticated cyber threats due to resource constraints and diverse traffic patterns. This project proposes a hybrid architecture combining **Long Short-Term Memory (LSTM)** networks for temporal feature extraction with robust ensemble classifiers for accurate final classification.

### Proposed Hybrid Models:
1. **H1: LSTM + XGBoost** (Deep feature extraction followed by Gradient Boosting classification)
2. **H2: LSTM + Random Forest** (Deep feature extraction followed by Random Forest classification)

---

## 📊 Experimental Results

A performance comparison between the two hybrid architectures yielded the following metrics:

| Metric | H1: LSTM + XGBoost | H2: LSTM + Random Forest |
| :--- | :--- | :--- |
| **Accuracy** | **90.45%** | 87.03% |
| **F1-Score** | **90.37%** | 86.90% |
| **Attack Recall** | **96.31%** | 3.73% |
| **ROC-AUC (Avg. Prec.)** | **0.9754** | 0.9518 |
| **Execution Time** | 282.4s | **217.8s** |

* **Key Takeaway:** The **LSTM + XGBoost** model achieves superior overall accuracy and exceptional attack detection capabilities (Recall of 96.31%), making it the optimal choice for secure IoT environments.

---

## 🛠️ Tech Stack & Tools
* **Language:** Python
* **Deep Learning / ML:** TensorFlow / Keras, Scikit-Learn, XGBoost
* **Data Processing:** Pandas, NumPy
* **Dataset:** UNSW-NB15 Benchmark Dataset

---

## ⚙️ How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/soufian-haskouk/IoT-Anomaly-Detection-LSTM-Ensemble.git](https://github.com/soufian-haskouk/IoT-Anomaly-Detection-LSTM-Ensemble.git)
