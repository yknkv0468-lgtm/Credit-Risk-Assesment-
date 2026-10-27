import pandas as pd
import numpy as np
import joblib
from scipy.stats import uniform
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier


print("Loading and cleaning data...")
df = pd.read_csv('credit_risk_dataset (1).csv')


df = df.drop_duplicates()
df = df[(df['person_age'] >= 18) & (df['person_age'] <= 100)]
df = df[(df["person_emp_length"] <= df['person_age']) & (df["person_emp_length"] <= 60)]
df = df[(df['loan_amnt'] >= 0) & (df['loan_int_rate'] >= 0)]

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
                'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

preprocessor_XG = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean"))]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value='missing')),
            ("encoder", OneHotEncoder(handle_unknown="ignore",sparse_output=False))
        ]), categorical_cols)
    ])
preprocessor_XG.set_output(transform="pandas")
constraints = {
    'num__person_income': -1,
    'num__person_emp_length': -1,
    'num__loan_amnt': 1,
    'num__loan_int_rate': 1,
    'num__loan_percent_income': 1
}

neg, pos = np.bincount(y_train)
weight_factor = neg / pos

xgb_crosval_pipeline = Pipeline([
    ("preprocessor", preprocessor_XG),
    ("classifier", XGBClassifier(booster='gbtree', n_estimators=300, max_depth=5, 
                                 learning_rate=0.1, 
                                 random_state=42, verbosity=0, n_jobs=-1,
                                 monotone_constraints=constraints,
                                 scale_pos_weight=weight_factor))
])


print("Running RandomizedSearchCV (this may take a minute)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'classifier__n_estimators': [100, 200],              
    'classifier__max_depth': [3, 5, 7],                  
    'classifier__learning_rate': [0.03, 0.1],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0],
    'classifier__min_child_weight': [1, 3, 5],
    'classifier__gamma': uniform(0, 5) 
}

random_search = RandomizedSearchCV(
    estimator=xgb_crosval_pipeline,
    param_distributions=param_grid,
    n_iter=10,
    scoring='average_precision',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
print("Best parameters found!")


print("Calibrating the best model...")
calibrated_model = CalibratedClassifierCV(
    best_model,
    method='isotonic',
    cv=5
)
calibrated_model.fit(X_train, y_train)


print("Calculating F1-Optimal Threshold...")
prob = best_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, prob)


f1 = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1[:-1])
best_threshold = float(thresholds[best_idx]) 

print(f"Optimal Threshold found: {best_threshold:.4f}")


joblib.dump(best_model, 'credit_risk_model.pkl')
joblib.dump(best_threshold, 'best_threshold.pkl')
print("Model and Threshold successfully saved! You can now run your Streamlit app.")