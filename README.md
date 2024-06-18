# Summer-Analytics
This repository is meant for summer analytics course assignments.

Project Objective
This project aims to predict the likelihood of individuals receiving the xyz and seasonal flu vaccines using machine learning techniques.
DataHack Project
Approach
Data Preparation:
Loaded and preprocessed features and labels from CSV files.
Imputed missing values and encoded categorical features.
Standardized numerical features for consistency.

Modeling:
Utilized RandomForestClassifier within a MultiOutputClassifier setup to predict probabilities for both vaccines simultaneously.
Implemented a pipeline to streamline preprocessing and modeling steps.

Hyperparameter Tuning:
Applied RandomizedSearchCV for optimizing model hyperparameters.
Evaluated model performance using 3-fold cross-validation.

Evaluation:
Assessed model performance using ROC AUC scores.
Achieved ROC AUC scores: xyz_vaccine = 0.8669, seasonal_vaccine = 0.8585.
Prediction and Submission:

Generated predictions for the test set.
Saved predictions in CSV format (submission.csv) for submission.

