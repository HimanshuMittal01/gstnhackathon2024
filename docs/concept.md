This document describes the overall approach and it is intended for the purpose of filling 'Idea / Concept' field in the submission form of the hackathon.

The given problem statement is a binary classification problem and data science methodologies seems appropriate for the analysis. Since, the data is already converted and given in the numeric type and additionally the column metadata is missing, it essentially boils down to a pure mathematical problem. Therefore, I invested most of the time in analyzing relationships between the columns and visualizing distributions to do good feature engineering.

Interestingly, the hackathon does not reveal the judging metric and leaves open choice to the participants. I decided to optimize for the **PR-AUC score** (also known as average precision) because PR curve focuses on achieving better predictions for the positive class and it helps in the model selection in situation of the unbalanced dataset.

My approach is to:
1. Setup the baseline score
2. Do Exploratory Data Analysis (EDA)
3. Handle missing values
4. Feature engineering
5. Model building and training
6. Evaluation and Hyperparameter tuning
7. Feature selection based on feature importance
8. Make submission docs.

