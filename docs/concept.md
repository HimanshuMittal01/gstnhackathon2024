# Idea / Concept
This document outlines the approach for addressing a binary classification problem in a hackathon submission. The problem involves numeric data with missing metadata, making it a mathematical challenge. The focus is on analyzing relationships between columns and visualizing distributions for effective feature engineering. Since the judging metric is undisclosed, the **PR-AUC score** (average precision) was chosen for optimization due to its focus on positive class predictions, suitable for imbalanced datasets.

The methodology includes:
1. Establishing a baseline score
2. Conducting Exploratory Data Analysis (EDA)
3. Handling missing values
4. Feature engineering
5. Model building and training
6. Hyperparameter tuning
7. Feature selection based on importance
8. Creating submission documents.

# Project Description

**Overview**:  
To predict the target variable, I developed a machine learning (ML) solution with a focus on effective feature engineering. I introduced indicator variables for missing and outlier values, and created new features by combining existing columns. The model is built using a stacked classifier of 10 tree-based models, with Logistic Regression serving as the final estimator to combine predictions from the base models.

**Key Observation**:  
A critical finding was that when 'Column18' equals zero or 'Column1' equals 2495, the target variable is consistently zero. This hypothesis was statistically validated and informed the decision to create diversity in the base estimators by selecting negative samples based on this pattern and adjusting hyperparameters.

**Feature Engineering**:  
To improve predictive performance, I focused on deriving meaningful features. This involved:
- Creating indicators for missing values, such as missingness patterns between 'Column3' and 'Column4'.
- Addressing outliers by using an isolation forest to classify them and add these as a new feature.

**Reproducibility**:  
The Python code for training and inference, along with notebooks for Exploratory Data Analysis (EDA), hyperparameter tuning, and feature importance, is available in the repository. The model’s performance, including the PR-AUC score of 0.940762, can be reproduced using this code.
