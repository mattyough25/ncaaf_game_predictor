import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Read Dataset for Training
file_path = "/Users/matthewyough/Desktop/Github/ncaaf_game_predictor"
file_name = "Model_Training_Dataset.xlsx"

# Model Output Path and File Names
model_path = "/Users/matthewyough/Desktop/Github/ncaaf_game_predictor"
home_model_file = "Home_Team_Model.pkl"
away_model_file = "Away_Team_Model.pkl"

data_df = pd.read_excel(f"{file_path}/{file_name}", index_col=0)
data_df = data_df.dropna(axis = 0).reset_index()

# Define Metrics for Input Variables
X = data_df[['rank', 'opponent_rank', 'off_success_rate','def_success_rate', 'def_per_game_ppa',
                   'off_per_game_ppa', 'passing_yards_per_game', 'rushing_yards_per_game', 'TDs_per_game', 'time_of_possession_per_game',
                   'off_turnovers_per_game', 'penalty_yards_per_game', 'sacks_per_game', 'def_interceptions_per_game', 'team_code', 'opp_code',
                    'opp_off_success_rate','opp_def_success_rate', 'opp_def_per_game_ppa','opp_off_per_game_ppa', 'opp_passing_yards_per_game', 
                    'opp_rushing_yards_per_game', 'opp_TDs_per_game', 'opp_time_of_possession_per_game', 'opp_off_turnovers_per_game', 
                    'opp_penalty_yards_per_game', 'opp_sacks_per_game', 'opp_def_interceptions_per_game']]
X = X.reset_index(drop=True)

# Split Data for Home and Away Models
X_team = X[['rank', 'opponent_rank', 'off_success_rate', 'off_per_game_ppa', 'passing_yards_per_game', 'rushing_yards_per_game', 'TDs_per_game', 
            'time_of_possession_per_game', 'off_turnovers_per_game', 'penalty_yards_per_game',  'team_code', 'opp_code',
            'opp_def_success_rate', 'opp_def_per_game_ppa', 'opp_time_of_possession_per_game','opp_penalty_yards_per_game', 
            'opp_sacks_per_game', 'opp_def_interceptions_per_game']]
X_opp = X[['rank', 'opponent_rank', 'def_success_rate', 'def_per_game_ppa','time_of_possession_per_game','penalty_yards_per_game', 'sacks_per_game',
            'def_interceptions_per_game', 'team_code', 'opp_code','opp_off_success_rate','opp_off_per_game_ppa', 'opp_passing_yards_per_game', 
            'opp_rushing_yards_per_game', 'opp_TDs_per_game', 'opp_time_of_possession_per_game', 'opp_off_turnovers_per_game', 
            'opp_penalty_yards_per_game']]

y_team = data_df['team_points']
y_opp = data_df['opponent_points']

# Home Linear Regression Model
X_train, X_test, y_train, y_test = train_test_split(X_team, y_team, test_size = 0.20)

regr_home = LinearRegression()
regr_home.fit(X_train, y_train.values.ravel())
print(regr_home.score(X_test, y_test))

# Away Linear Regression model
X_train, X_test, y_train, y_test = train_test_split(X_opp, y_opp, test_size = 0.20)

regr_away = LinearRegression()
regr_away.fit(X_train, y_train.values.ravel())
print(regr_away.score(X_test, y_test))

# Save Models
joblib.dump(regr_home, f"{model_path}/{home_model_file}")
joblib.dump(regr_away, f"{model_path}/{away_model_file}")