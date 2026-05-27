import pandas as pd
import os

# =========================================================
# LOAD DATA
# =========================================================

matches = pd.read_csv("data/processed/matches_cleaned.csv")
deliveries = pd.read_csv("data/processed/deliveries_cleaned.csv")

print("Datasets Loaded!\n")

# =========================================================
# SHOW COLUMNS
# =========================================================

print("MATCHES COLUMNS:")
print(matches.columns)

print("\nDELIVERIES COLUMNS:")
print(deliveries.columns)

# =========================================================
# CREATE OUTPUT DIRECTORY
# =========================================================

os.makedirs("data/processed/advanced_features", exist_ok=True)

# =========================================================
# DETECT RUN COLUMN AUTOMATICALLY
# =========================================================

possible_run_columns = [
    'batsman_runs',
    'batter_runs',
    'runs_off_bat',
    'runs_of_bat',
    'total_runs',
    'runs'
]

RUN_COLUMN = None

for col in possible_run_columns:
    if col in deliveries.columns:
        RUN_COLUMN = col
        break

if RUN_COLUMN is None:
    raise Exception(
        "No valid runs column found in deliveries dataset!"
    )

print(f"\nUsing Run Column: {RUN_COLUMN}")

# =========================================================
# RECENT FORM SCORE
# =========================================================

recent_matches = matches.sort_values(by='date')

team_form = []

teams = pd.concat([
    recent_matches['team1'],
    recent_matches['team2']
]).unique()

for team in teams:

    team_games = recent_matches[
        (recent_matches['team1'] == team) |
        (recent_matches['team2'] == team)
    ]

    recent_5 = team_games.tail(5)

    wins = (
        recent_5['match_winner'] == team
    ).sum()

    form_score = wins / 5

    team_form.append({
        "team": team,
        "recent_form_score": form_score
    })

team_form_df = pd.DataFrame(team_form)

print("Recent Form Feature Created!")

# =========================================================
# CHASE SUCCESS RATE
# =========================================================

chasing_matches = matches[
    matches['toss_decision'] == 'field'
]

chase_success = chasing_matches.groupby(
    'match_winner'
).size().reset_index(name='successful_chases')

print("Chase Success Feature Created!")

# =========================================================
# POWERPLAY EFFICIENCY
# =========================================================

powerplay = deliveries[
    deliveries['over'] <= 6
]

powerplay_runs = powerplay.groupby(
    'batting_team'
)[RUN_COLUMN].mean().reset_index()

powerplay_runs.columns = [
    'team',
    'avg_powerplay_runs'
]

print("Powerplay Feature Created!")

# =========================================================
# DEATH OVER EFFICIENCY
# =========================================================

death_overs = deliveries[
    deliveries['over'] >= 16
]

death_runs = death_overs.groupby(
    'batting_team'
)[RUN_COLUMN].mean().reset_index()

death_runs.columns = [
    'team',
    'avg_death_overs_runs'
]

print("Death Overs Feature Created!")

# =========================================================
# SAVE FEATURES
# =========================================================

team_form_df.to_csv(
    "data/processed/advanced_features/recent_form.csv",
    index=False
)

chase_success.to_csv(
    "data/processed/advanced_features/chase_success.csv",
    index=False
)

powerplay_runs.to_csv(
    "data/processed/advanced_features/powerplay.csv",
    index=False
)

death_runs.to_csv(
    "data/processed/advanced_features/death_overs.csv",
    index=False
)

print("\nAdvanced Feature Engineering Completed Successfully!")