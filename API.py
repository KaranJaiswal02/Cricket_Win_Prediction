from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load Dataset
data = pd.read_csv('odi_Matches_Data.csv')  # Update with correct path

# Encode Teams & Venue
target_column = "Match Venue (Country)"  # Defaulting to country
le_team = LabelEncoder()
le_venue = LabelEncoder()

all_teams = pd.concat([data['team1'], data['team2'], data['toss_winner'], data['match_winner']]).unique()
le_team.fit(all_teams)
data['team1'] = le_team.transform(data['team1'])
data['team2'] = le_team.transform(data['team2'])
data['toss_winner'] = le_team.transform(data['toss_winner'])
data['match_winner'] = le_team.transform(data['match_winner'])

data['venue'] = le_venue.fit_transform(data[target_column])

data['home_advantage'] = np.where(data['venue'] == data['team1'], 1, 0)
data['team1_strength'] = data.groupby('team1')['match_winner'].transform('mean')
data['team2_strength'] = data.groupby('team2')['match_winner'].transform('mean')
data['recent_form1'] = data.groupby('team1')['match_winner'].transform(lambda x: x.rolling(5, min_periods=1).mean())
data['recent_form2'] = data.groupby('team2')['match_winner'].transform(lambda x: x.rolling(5, min_periods=1).mean())

data['match_winner'] = (data['match_winner'] == data['team1']).astype(int)

features = ['home_advantage', 'toss_winner', 'team1_strength', 'team2_strength', 'recent_form1', 'recent_form2']
target = 'match_winner'
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
class_ratio = data['match_winner'].value_counts()[0] / data['match_winner'].value_counts()[1]

model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, scale_pos_weight=class_ratio)
model.fit(X_train, y_train)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_json = request.json  # Rename to avoid confusion with global 'data'
        
        team1_name = data_json['team1']
        team2_name = data_json['team2']
        venue_name = data_json['venue']
        toss_winner_name = data_json['toss_winner']

        # Check if teams and venue exist in the encoder
        if team1_name not in le_team.classes_:
            return jsonify({"error": f"Invalid team name provided: {team1_name}"}), 400
        if team2_name not in le_team.classes_:
            return jsonify({"error": f"Invalid team name provided: {team2_name}"}), 400
        if venue_name not in le_venue.classes_:
            return jsonify({"error": f"Invalid venue provided: {venue_name}"}), 400

        # Convert to numerical values using label encoders
        team1 = le_team.transform([team1_name])[0]
        team2 = le_team.transform([team2_name])[0]
        venue = le_venue.transform([venue_name])[0]
        home_advantage = 1 if venue == team1 else 0
        toss_winner = 1 if toss_winner_name == team1_name else 0

        # Ensure these values are correctly retrieved
        team1_strength = data[data['team1'] == team1]['team1_strength'].mean()
        team2_strength = data[data['team2'] == team2]['team2_strength'].mean()
        recent_form1 = data[data['team1'] == team1]['recent_form1'].mean()
        recent_form2 = data[data['team2'] == team2]['recent_form2'].mean()

        if pd.isna(team1_strength) or pd.isna(team2_strength) or pd.isna(recent_form1) or pd.isna(recent_form2):
            return jsonify({"error": "Missing data for selected teams. Please try different teams."}), 400

        # Prepare input data
        input_data = np.array([[team1_strength, team2_strength, home_advantage, toss_winner, recent_form1, recent_form2]])

        # Predict probability
        probability = model.predict_proba(input_data)[0][1]

        return jsonify({"team1": team1_name, "team2": team2_name, "win_probability": probability * 100})

    except Exception as e:
        print("Error:", str(e))  # Print error in console
        return jsonify({"error": str(e)}), 500  # Return actual error message


if __name__ == '__main__':
      # Run the app on port 5000
    app.run(debug=True, port=5000)