import pandas as pd
import os

# =========================================================
# LOAD MAIN MATCH DATA
# =========================================================

matches = pd.read_csv(
    "data/processed/matches_cleaned.csv"
)

print("Main Match Dataset Loaded!")

# =========================================================
# LOAD FEATURE FILES
# =========================================================

team_win = pd.read_csv(
    "data/processed/features/team_win_percentage.csv"
)

recent_form = pd.read_csv(
    "data/processed/advanced_features/recent_form.csv"
)

powerplay = pd.read_csv(
    "data/processed/advanced_features/powerplay.csv"
)

death_overs = pd.read_csv(
    "data/processed/advanced_features/death_overs.csv"
)

print("Feature Files Loaded!")

# =========================================================
# CREATE COPY
# =========================================================

dataset = matches.copy()

# =========================================================
# TEAM 1 FEATURES
# =========================================================

dataset = dataset.merge(
    team_win,
    left_on='team1',
    right_on='team',
    how='left'
)

dataset.rename(
    columns={
        'win_percentage': 'team1_win_percentage'
    },
    inplace=True
)

dataset.drop(columns=['team'], inplace=True)

# =========================================================
# TEAM 2 FEATURES
# =========================================================

dataset = dataset.merge(
    team_win,
    left_on='team2',
    right_on='team',
    how='left'
)

dataset.rename(
    columns={
        'win_percentage': 'team2_win_percentage'
    },
    inplace=True
)

dataset.drop(columns=['team'], inplace=True)

# =========================================================
# RECENT FORM - TEAM 1
# =========================================================

dataset = dataset.merge(
    recent_form,
    left_on='team1',
    right_on='team',
    how='left'
)

dataset.rename(
    columns={
        'recent_form_score': 'team1_recent_form'
    },
    inplace=True
)

dataset.drop(columns=['team'], inplace=True)

# =========================================================
# RECENT FORM - TEAM 2
# =========================================================

dataset = dataset.merge(
    recent_form,
    left_on='team2',
    right_on='team',
    how='left'
)

dataset.rename(
    columns={
        'recent_form_score': 'team2_recent_form'
    },
    inplace=True
)

dataset.drop(columns=['team'], inplace=True)

# =========================================================
# POWERPLAY FEATURES
# =========================================================

dataset = dataset.merge(
    powerplay,
    left_on='team1',
    right_on='team',
    how='left'
)

dataset.rename(
    columns={
        'avg_powerplay_runs': 'team1_powerplay'
    },
    inplace=True
)

dataset.drop(columns=['team'], inplace=True)

dataset = dataset.merge(
    powerplay,
    left_on='team2',
    right_on='team',
    how='left'
)

dataset.rename(
    columns={
        'avg_powerplay_runs': 'team2_powerplay'
    },
    inplace=True
)

dataset.drop(columns=['team'], inplace=True)

# =========================================================
# DEATH OVERS FEATURES
# =========================================================

dataset = dataset.merge(
    death_overs,
    left_on='team1',
    right_on='team',
    how='left'
)

dataset.rename(
    columns={
        'avg_death_overs_runs': 'team1_death_overs'
    },
    inplace=True
)

dataset.drop(columns=['team'], inplace=True)

dataset = dataset.merge(
    death_overs,
    left_on='team2',
    right_on='team',
    how='left'
)

dataset.rename(
    columns={
        'avg_death_overs_runs': 'team2_death_overs'
    },
    inplace=True
)

dataset.drop(columns=['team'], inplace=True)

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

dataset.fillna(0, inplace=True)

# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs(
    "data/final",
    exist_ok=True
)

# =========================================================
# SAVE FINAL DATASET
# =========================================================

dataset.to_csv(
    "data/final/master_training_dataset.csv",
    index=False
)

print("\nMaster Training Dataset Created Successfully!")

print("\nFinal Dataset Shape:")
print(dataset.shape)

print("\nColumns:")
print(dataset.columns)