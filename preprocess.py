import pandas as pd
import os

# ---------------- FILE PATHS ----------------

MATCHES_PATH = "data/raw/historical/archive_data/matches.csv"
DELIVERIES_PATH = "data/raw/historical/archive_data/deliveries.csv"

OUTPUT_DIR = "data/processed"

# ---------------- LOAD DATA ----------------

print("Loading datasets...")

matches = pd.read_csv(MATCHES_PATH)
deliveries = pd.read_csv(DELIVERIES_PATH)

print("Datasets loaded successfully!")

# ---------------- BASIC INFO ----------------

print("Matches Shape:", matches.shape)
print("Deliveries Shape:", deliveries.shape)

# ---------------- REMOVE DUPLICATES ----------------

matches.drop_duplicates(inplace=True)
deliveries.drop_duplicates(inplace=True)

# ---------------- HANDLE MISSING VALUES ----------------

matches.fillna("Unknown", inplace=True)
deliveries.fillna(0, inplace=True)

# ---------------- STANDARDIZE TEAM NAMES ----------------

team_replacements = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings"
}

matches.replace(team_replacements, inplace=True)
deliveries.replace(team_replacements, inplace=True)

# ---------------- CREATE OUTPUT DIRECTORY ----------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- SAVE CLEANED FILES ----------------

matches.to_csv(f"{OUTPUT_DIR}/matches_cleaned.csv", index=False)
deliveries.to_csv(f"{OUTPUT_DIR}/deliveries_cleaned.csv", index=False)

print("Cleaned datasets saved successfully!")
print("Preprocessing Completed!")