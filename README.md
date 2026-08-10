# 🧬 Nature-Inspired Feature Selection for Breast Cancer Diagnosis

An academic collaborative mini-project focused on optimizing high-dimensional medical data classification using nature-inspired metaheuristic optimization algorithms combined with Machine Learning.

---

## 📋 Project Overview
High-dimensional datasets often contain redundant or noisy features that degrade classifier performance. This project applies nature-inspired optimization algorithms to search through the combinatorial feature space ($2^{30}$) of the **Breast Cancer Wisconsin (Diagnostic) Dataset**, identifying the smallest and most effective subset of features while maintaining high diagnostic accuracy.

### **Key Objectives**
* **Establish a Baseline:** Train and evaluate a K-Nearest Neighbors (KNN) classifier using all 30 original features.
* **Implement Metaheuristic Optimizers:** Use bio-inspired algorithms to perform wrapper-based feature selection:
  * **Genetic Algorithm (GA)**
  * **Particle Swarm Optimization (PSO)**
  * **Grey Wolf Optimizer (GWO)**
* **Deploy an Interactive Web Application:** Build a live Streamlit app allowing users to input patient tumor records and receive instant diagnostic predictions.

---

## 👥 Team Responsibilities & Workflow
* **ITBIN-2313-0056 (Data Engineer & Baseline Developer):** Data cleaning, exploratory data analysis, data preprocessing/scaling, and baseline KNN implementation across all 30 features.
* **ITBIN-2313-0023 (GA Specialist):** Implementation and tuning of the Genetic Algorithm for binary feature subset selection.
* **ITBNM-2313-0024 (PSO Specialist):** Implementation and tuning of Particle Swarm Optimization for feature reduction.
* **ITBNM-2313-0061 (GWO Specialist & Evaluator):** Implementation of the Grey Wolf Optimizer and master comparative analysis (Accuracy, Precision, Recall, F1-Score, and Feature Count).

---

## 📊 Dataset Information
* **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset (`brca.csv`)
* **Samples:** 569 instances
* **Features:** 30 numerical tumor measurement attributes (e.g., radius, texture, perimeter, area, smoothness)
* **Target Classes:** Binary diagnosis (`M` = Malignant, `B` = Benign)

---

## 🚀 Getting Started Locally

### **1. Clone the Repository**
```bash
git clone [https://github.com/n3cx011/Nature-Inspired-Feature-Selection-For-Breast-Cancer-Diagnosis.git](https://github.com/n3cx011/Nature-Inspired-Feature-Selection-For-Breast-Cancer-Diagnosis.git)
cd Nature-Inspired-Feature-Selection-For-Breast-Cancer-Diagnosis