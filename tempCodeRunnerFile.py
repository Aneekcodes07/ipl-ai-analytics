import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# =========================================================
# LOAD DATA
# =========================================================

matches = pd.read_csv(
    "data/processed/matches_cleaned.csv"
)

print("Dataset Loaded!")

# =========================================================
# SELECT IMPORTANT COLUMNS
# =========================================================

model_data = matches[[
    'team1',
    'team2',
    'venue',
    'toss_winner',
    'toss_decision',
    'match_winner'
]]

print("\nSelected ML Columns!")

# =========================================================
# REMOVE MISSING VALUES
# =========================================================

model_data.dropna(inplace=True)

# =========================================================
# ENCODE CATEGORICAL FEATURES
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
# LOGISTIC REGRESSION
# =========================================================

lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_accuracy = accuracy_score(
    y_test,
    lr_predictions
)

print("\nLogistic Regression Accuracy:")
print(lr_accuracy)

# =========================================================
# RANDOM FOREST
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

# =========================================================
# MODEL COMPARISON
# =========================================================

print("\nMODEL COMPARISON")

print(f"Logistic Regression: {lr_accuracy:.4f}")
print(f"Random Forest: {rf_accuracy:.4f}")

# =========================================================
# BEST MODEL
# =========================================================

if rf_accuracy > lr_accuracy:
    best_model = "Random Forest"
else:
    best_model = "Logistic Regression"

print(f"\nBest Model: {best_model}")

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        rf_predictions
    )
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_test,
        rf_predictions
    )
)

print("\nML Training Pipeline Completed Successfully!")