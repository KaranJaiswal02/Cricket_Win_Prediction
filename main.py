import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv('odi_Matches_Data.csv')  # Update with correct path


# User input for venue type
venue_type = input("Would you like to base predictions on 'city', 'stadium', or 'country'? ").strip().lower()

venue_mapping = {
    "stadium": "Match Venue (Stadium)",
    "city": "Match Venue (City)",
    "country": "Match Venue (Country)"
}

venue_column = venue_mapping.get(venue_type, "Match Venue (Country)")

if venue_column not in data.columns:
    raise KeyError(f"'{venue_column}' column not found in dataset. Available columns: {data.columns}")


print(data[['match_winner', 'team1', 'team2']].head(20))  # BEFORE encoding


import numpy as np

# Shuffle team1 and team2 randomly
random_mask = np.random.rand(len(data)) > 0.5
data.loc[random_mask, ['team1', 'team2']] = data.loc[random_mask, ['team2', 'team1']].values

# Update match_winner after swapping
def update_winner(row):
    if row['match_winner'] == row['team1']:
        return row['team1']
    elif row['match_winner'] == row['team2']:
        return row['team2']
    return row['match_winner']

data.loc[random_mask, 'match_winner'] = data[random_mask].apply(update_winner, axis=1)



# Ensure all relevant columns are strings before encoding
data['team1'] = data['team1'].astype(str)
data['team2'] = data['team2'].astype(str)
data['toss_winner'] = data['toss_winner'].astype(str)
data['match_winner'] = data['match_winner'].astype(str)
data[venue_column] = data[venue_column].astype(str)  # Ensure venue column is string

# Initialize LabelEncoders
le_team = LabelEncoder()
le_venue = LabelEncoder()

# ✅ Fit LabelEncoder on ALL unique team names (avoids unseen label errors)
all_teams = pd.concat([data['team1'], data['team2'], data['toss_winner'], data['match_winner']]).unique()
le_team.fit(all_teams)  # Now contains all possible teams

# ✅ Encode teams correctly
data['team1'] = le_team.transform(data['team1'])
data['team2'] = le_team.transform(data['team2'])
data['toss_winner'] = le_team.transform(data['toss_winner'])
data['match_winner'] = le_team.transform(data['match_winner'])

# ✅ Encode venue correctly
data['venue'] = le_venue.fit_transform(data[venue_column])

# ✅ Feature Engineering
data['home_advantage'] = np.where(data['venue'] == data['team1'], 1, 0)
data['team1_strength'] = data.groupby('team1')['match_winner'].transform('mean')
data['team2_strength'] = data.groupby('team2')['match_winner'].transform('mean')
data['recent_form1'] = data.groupby('team1')['match_winner'].transform(lambda x: x.rolling(5, min_periods=1).mean())
data['recent_form2'] = data.groupby('team2')['match_winner'].transform(lambda x: x.rolling(5, min_periods=1).mean())

# ✅ Ensure match_winner is binary (1 if team1 wins, else 0)
data['match_winner'] = (data['match_winner'] == data['team1']).astype(int)

print("✅ Encoding Completed Successfully!")


print(data['match_winner'].value_counts())



# Selecting Features and Target
features = ['home_advantage', 'toss_winner', 'team1_strength', 'team2_strength', 'recent_form1', 'recent_form2']
target = 'match_winner'

X = data[features]
y = data[target]

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class_ratio = data['match_winner'].value_counts()[0] / data['match_winner'].value_counts()[1]
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, scale_pos_weight=class_ratio)

model.fit(X_train, y_train)


# Function to Predict Match Outcome

def predict_match(team1_name, team2_name, venue_name, toss_winner):
    if team1_name not in le_team.classes_ or team2_name not in le_team.classes_:
        raise ValueError("Invalid team name provided.")

    if venue_name not in le_venue.inverse_transform(range(len(le_venue.classes_))):
        raise ValueError("Invalid venue provided.")

    team1 = le_team.transform([team1_name])[0]
    team2 = le_team.transform([team2_name])[0]
    venue = le_venue.transform([venue_name])[0]
    
    home_advantage = 1 if venue == team1 else 0
    toss_winner = 1 if toss_winner == team1 else 0

    # Use mean instead of zero for missing values
    overall_team_strength = data[['team1_strength', 'team2_strength']].mean().mean()
    overall_recent_form = data[['recent_form1', 'recent_form2']].mean().mean()

    team1_strength = data[data['team1'] == team1]['team1_strength'].mean()
    team2_strength = data[data['team2'] == team2]['team2_strength'].mean()
    recent_form1 = data[data['team1'] == team1]['recent_form1'].mean()
    recent_form2 = data[data['team2'] == team2]['recent_form2'].mean()

    # Replace NaN with mean instead of zero
    team1_strength = overall_team_strength if np.isnan(team1_strength) else team1_strength
    team2_strength = overall_team_strength if np.isnan(team2_strength) else team2_strength
    recent_form1 = overall_recent_form if np.isnan(recent_form1) else recent_form1
    recent_form2 = overall_recent_form if np.isnan(recent_form2) else recent_form2

    # Corrected input_data format
    input_data = np.array([[team1_strength, team2_strength, home_advantage, toss_winner, recent_form1, recent_form2]])

    probability = model.predict_proba(input_data)[0][1]  # Probability of team1 winning
    print(f'Probability of {team1_name} Winning: {probability * 100:.2f}%')
    return probability


predict_match("New Zealand", "Australia", "Melbourne", "Australia")