import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score

# Load datasets
train_features = pd.read_csv(r"C:\Users\Nandini Manoj Pathak\Downloads\training_set_features.csv")
train_labels = pd.read_csv(r"C:\Users\Nandini Manoj Pathak\Downloads\training_set_labelscsv.csv")
test_features = pd.read_csv(r"C:\Users\Nandini Manoj Pathak\Downloads\test_set_features.csv")

# Drop respondent_id column
X = train_features.drop(columns=['respondent_id'])
y = train_labels.drop(columns=['respondent_id'])
X_test = test_features.drop(columns=['respondent_id'])

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


model = MultiOutputClassifier(RandomForestClassifier(random_state=42))


pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)
                          ])


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'classifier__estimator__n_estimators': [100, 200],
    'classifier__estimator__max_depth': [None, 10, 20],
    'classifier__estimator__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)


y_val_pred_proba = grid_search.best_estimator_.predict_proba(X_val)


roc_auc_xyz = roc_auc_score(y_val['xyz_vaccine'], y_val_pred_proba[0][:, 1])
roc_auc_seasonal = roc_auc_score(y_val['seasonal_vaccine'], y_val_pred_proba[1][:, 1])
mean_roc_auc = np.mean([roc_auc_xyz, roc_auc_seasonal])

print(f'ROC AUC Score for xyz_vaccine: {roc_auc_xyz}')
print(f'ROC AUC Score for seasonal_vaccine: {roc_auc_seasonal}')
print(f'Mean ROC AUC Score: {mean_roc_auc}')

# Predict on test set
y_test_pred_proba = grid_search.best_estimator_.predict_proba(X_test)

# Prepare the submission
print("Submission file created: submission.csv")

print("Current Working Directory:", os.getcwd())

# Create a DataFrame for the submission
submission = pd.DataFrame({
    'respondent_id': test_features['respondent_id'],
    'xyz_vaccine': y_test_pred_proba[0][:, 1],
    'seasonal_vaccine': y_test_pred_proba[1][:, 1]
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")