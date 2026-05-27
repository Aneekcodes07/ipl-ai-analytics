import pandas as pd
import os

# ---------------- LOAD CLEANED DATA ----------------

matches = pd.read_csv("data/processed/matches_cleaned.csv")
deliveries = pd.read_csv("data/processed/deliveries_cleaned.csv")

print("Datasets Loaded!\n")

# ---------------- SHOW COLUMNS ----------------

print("Matches Columns:")
print(matches.columns)

print("\nDeliveries Columns:")
print(deliveries.columns)

# ---------------- TEAM WIN PERCENTAGE ----------------

team_matches = matches["match_winner"].value_counts()

total_matches = pd.concat([
    matches['team1'],
    matches['team2']
]).value_counts()

team_win_percentage = (
    (team_matches / total_matches) * 100
).fillna(0)

team_win_df = pd.DataFrame({
    "team": team_win_percentage.index,
    "win_percentage": team_win_percentage.values
})

print("\nTeam Win Percentage Created!")

# ---------------- TOSS IMPACT ----------------

matches['toss_win_match_win'] = (
    matches['toss_winner'] == matches['match_winner']
).astype(int)

toss_impact = matches.groupby('toss_winner')[
    'toss_win_match_win'
].mean().reset_index()

toss_impact.columns = [
    'team',
    'toss_win_match_win_rate'
]

print("Toss Impact Feature Created!")

# ---------------- VENUE ADVANTAGE ----------------

venue_win = matches.groupby(
    ['venue', 'match_winner']
).size().reset_index(name='wins')

print("Venue Advantage Feature Created!")

# ---------------- CREATE OUTPUT DIRECTORY ----------------

os.makedirs("data/processed/features", exist_ok=True)

# ---------------- SAVE FEATURES ----------------

team_win_df.to_csv(
    "data/processed/features/team_win_percentage.csv",
    index=False
)

toss_impact.to_csv(
    "data/processed/features/toss_impact.csv",
    index=False
)

venue_win.to_csv(
    "data/processed/features/venue_win.csv",
    index=False
)

print("\nFeature files saved successfully!")

print("\nFeature Engineering Completed Successfully!")