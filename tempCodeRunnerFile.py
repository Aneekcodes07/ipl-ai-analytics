import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score

# =========================================================
# LOAD MASTER DATASET
# =========================================================

dataset = pd.read_csv(
    "data/final/master_training_dataset.csv"
)

print("Master Dataset Loaded!")

# =========================================================
# SELECT FEATURES
# =========================================================

model_data = dataset[[
    'team1',
    'team2',
    'venue',
    'toss_winner',
    'toss_decision',
    'team1_win_percentage',
    'team2_win_percentage',
    'team1_recent_form',
    'team2_recent_form',
    'team1_powerplay',
    'team2_powerplay',
    'team1_death_overs',
    'team2_death_overs',
    'match_winner'
]].copy()

print("\nSelected Advanced Features!")

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

model_data.fillna(0, inplace=True)

# =========================================================
# LABEL ENCODING
# =========================================================

label_encoders = {}

categorical_columns = [
    'team1',
    'team2',
    'venue',
    'toss_winner',
    'toss_decision',
    'match_winner'
]

for col in categorical_columns:

    le = LabelEncoder()

    model_data[col] = le.fit_transform(
        model_data[col]
    )

    label_encoders[col] = le

print("\nCategorical Encoding Completed!")

# =========================================================
# FEATURES & TARGET
# =========================================================

X = model_data.drop(
    'match_winner',
    axis=1
)

y = model_data['match_winner']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-Test Split Completed!")

# =========================================================
# MODELS
# =========================================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        ),

    "LightGBM":
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        ),

    "CatBoost":
        CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            verbose=0
        )
}

# =========================================================
# TRAINING LOOP
# =========================================================

results = {}

best_model = None
best_accuracy = 0

print("\nTraining Models...\n")

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    print(f"{name} Accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# =========================================================
# BEST MODEL
# =========================================================

print("\n==============================")
print(f"BEST MODEL: {best_model_name}")
print(f"BEST ACCURACY: {best_accuracy:.4f}")
print("==============================")

# =========================================================
# SAVE BEST MODEL
# =========================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest Model Saved Successfully!")

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

if hasattr(best_model, "feature_importances_"):

    feature_importance = pd.DataFrame({

        'Feature': X.columns,
        'Importance': best_model.feature_importances_

    })

    feature_importance = feature_importance.sort_values(
        by='Importance',
        ascending=False
    )

    print("\nTop Feature Importance:\n")

    print(feature_importance)

    feature_importance.to_csv(
        "models/feature_importance.csv",
        index=False
    )

print("\nAdvanced ML Training Completed Successfully!")