# Machine Learning & Deep Learning Projects

Welcome to my machine learning and deep learning journey! This repository contains hands-on projects where I explore, implement, and compare various ML and DL techniques on real-world datasets. Each notebook represents a different challenge and learning milestone.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [1. Diabetes Prediction](#1-diabetes-prediction)
- [2. Titanic Survival Prediction](#2-titanic-survival-prediction)
- [3. Telco Customer Churn (ML & DL)](#3-telco-customer-churn-ml--dl)
- [Key Learnings](#key-learnings)
- [How to Run](#how-to-run)
- [Future Plans](#future-plans)
- [Acknowledgements](#acknowledgements)

---

## Overview
This repository documents my progress in mastering machine learning and deep learning. I have tackled different datasets and problems, focusing on:
- Data preprocessing and cleaning
- Feature engineering and selection
- Model building, evaluation, and comparison
- Visualization and interpretation

## Project Structure
```
machine_learning/
├── diabetes_prediction.ipynb         # Predicting diabetes using SVM
├── titanic_survival_prediction.ipynb # Predicting Titanic survival using SVM
├── ML-DL.ipynb                      # Telco churn: feature selection, ML & DL
├── ML-DL.txt                        # Notes and learnings
├── README.md                        # This file
```

## 1. Diabetes Prediction
- **Notebook:** `diabetes_prediction.ipynb`
- **Goal:** Predict whether a patient has diabetes based on medical features.
- **Techniques:** Data exploration, standardization, SVM classification, accuracy evaluation, and prediction for new data.
- **Key Steps:**
  - Data loading and exploration
  - Feature scaling
  - Train-test split
  - SVM model training and evaluation
  - Making predictions for new patients

## 2. Titanic Survival Prediction
- **Notebook:** `titanic_survival_prediction.ipynb`
- **Goal:** Predict passenger survival on the Titanic.
- **Techniques:** Data cleaning, encoding, feature selection, SVM classification, and accuracy evaluation.
- **Key Steps:**
  - Handling missing values and encoding categorical data
  - Feature selection
  - Model training and evaluation
  - Survival prediction for new passengers

## 3. Telco Customer Churn (ML & DL)
- **Notebook:** `ML-DL.ipynb`
- **Goal:** Predict customer churn using both machine learning and deep learning models.
- **Techniques:**
  - Data preprocessing, encoding, and normalization
  - Feature selection (chi-squared, wrapper, random forest)
  - Model comparison: Logistic Regression, Random Forest, Gradient Boosting, SVM, k-NN
  - Deep learning with TensorFlow/Keras
  - Visualization of feature importance and model results
- **Key Steps:**
  - Data cleaning and encoding
  - Feature scaling (MinMax, Standard, Robust)
  - Feature selection and visualization
  - Model training, evaluation, and comparison
  - Building and evaluating a neural network

## Key Learnings
- The importance of data preprocessing and feature engineering
- How to select and compare features and models
- Practical experience with both classical ML and modern DL
- Visualization for better understanding and communication

## How to Run
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd machine_learning
   ```
2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn tensorflow missingno
   ```
3. Open any notebook in Jupyter or VS Code and run the cells sequentially.

## Future Plans
- Add more datasets and advanced models
- Experiment with hyperparameter tuning and model deployment
- Add more visualizations and explanations

## Acknowledgements
- Kaggle and UCI for datasets
- Scikit-learn, pandas, TensorFlow, and the open-source community

---

This repository is a reflection of my growth and curiosity in the field of data science. I hope it helps and inspires others on a similar path!