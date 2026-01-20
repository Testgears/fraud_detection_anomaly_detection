# Fraud Detection Using Anomaly Detection Techniques

This project explores multiple **anomaly detection approaches** to identify fraudulent transactions in an e-commerce setting. The objective is to compare global, local, and representation-based methods and analyze their strengths, limitations, and complementarity in detecting fraud under **class imbalance** and **limited labeled data**.

---

## Project Overview

Fraud detection is a challenging real-world problem due to:
- Highly imbalanced data
- Evolving and sophisticated fraud patterns
- Limited availability of labeled fraud examples

To address these challenges, this project evaluates several **unsupervised and semi-supervised anomaly detection techniques**, ranging from statistical baselines to deep learning models.

---

## Dataset

The dataset consists of transaction-level data from an e-commerce platform.  
Each observation corresponds to a **single product transaction** performed by a customer and includes transactional, behavioral, and customer-related features.

The target variable `fraude` indicates whether a transaction is fraudulent (1) or legitimate (0).

---

## Methods Implemented

### 1. Statistical Baseline (Z-score)
- Detects anomalies based on univariate deviation
- Highly interpretable but limited to extreme values

### 2. Clustering + Mahalanobis Distance
- Uses K-Means clustering to model local structure
- Anomalies identified via multivariate distance within clusters
- Sensitive to cluster assumptions and parameter choices

### 3. Autoencoder-Based Anomaly Detection
- Deep learning model trained to reconstruct normal transaction behavior
- Anomalies detected via high reconstruction error
- Includes:
  - Feature preprocessing
  - Threshold optimization using Precision–Recall and F1-score
  - Ablation study to assess feature importance

### 4. Local Outlier Factor (LOF)
- Density-based method detecting **local anomalies**
- Effective at identifying isolated observations
- High precision but limited recall in this dataset

---

## Evaluation Strategy

Model performance is evaluated using:
- Precision, Recall, and F1-score
- Confusion matrices
- Reconstruction error distributions
- Precision–Recall curves for threshold selection

An **ablation study** is conducted on the autoencoder to quantify the contribution of individual features.

---

## Key Findings

- The **autoencoder** provides the best overall balance between precision and recall.
- **LOF** achieves perfect precision but misses many fraud cases.
- **Clustering + Mahalanobis** improves over simple statistical baselines but remains limited.
- Different methods detect **different subsets of outliers**, reflecting complementary definitions of abnormality.

---
``` markdown
## Project Structure
``` text
.
├── DATA/
│   └── transactions_ecommerce.csv
├── README.md
└── fraud_detection_anomaly_detection.ipynb

---

## Technologies Used

- Python
- NumPy, Pandas, Scikit-learn
- TensorFlow / Keras
- Matplotlib, Seaborn, Plotly

---

## Key Takeaway

There is no single “best” anomaly detection method for fraud detection.  
Each approach captures a different notion of abnormality, and **combining complementary methods** leads to more robust fraud detection systems.

---

## Future Work

- Ensemble-based anomaly detection
- Semi-supervised learning with partial fraud labels
- Temporal modeling of transaction sequences
- Deployment-oriented threshold calibration
