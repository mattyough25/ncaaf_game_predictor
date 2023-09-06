import pandas as pd
import time
import cfbd
from cfbd.rest import ApiException
import numpy as np

# File Names and Paths
full_path = "/Users/matthewyough/Desktop/Github/ncaaf_game_predictor"
norm_path = "/Users/matthewyough/Desktop/Github/ncaaf_game_predictor"

full_file = "Full Dataset.xlsx"
norm_file = "Model_Training_Dataset.xlsx"

# Years to Scrape Data From
years = list(range(2016,2023))
years = years[::-1]

# Create API Connection
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = 'aqcJD+pZ9qCycPYBD3GrrjbRKnCLl95iAwLm8R6P5BIMaFHm8Swj6N1/+kTXG6eo'
configuration.api_key_prefix['Authorization'] = 'Bearer'

# API Instances
matchup_api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))
rankings_api_instance = cfbd.RankingsApi(cfbd.ApiClient(configuration))
teams_api_instance = cfbd.TeamsApi(cfbd.ApiClient(configuration))
advanced_stats_api_instance = cfbd.StatsApi(cfbd.ApiClient(configuration))
stats_api_instance = cfbd.StatsApi(cfbd.ApiClient(configuration))

# Create Team Dictionary
teams = teams_api_instance.get_teams()

# Dictionary Definitions
match_dict = {'team': [], 'opponent': [], 'team_points':[], 'opponent_points': [], 'week': [], 'season':[]}
rank ={'team':[], 'rank': [], 'week': [], 'season':[]}
advanced_dict = {'team': [], 'season': [], 'off_total_ppa':[], 'off_success_rate': [], 
                'def_total_ppa':[], 'def_success_rate': [], 'def_total_yards': []}

for year in years:
    print("Gathering " + str(year) + " Data")
    matchups = matchup_api_instance.get_games(year=year, division='FBS')

    # Make Matchup DF
    for matchup in matchups:
        match_dict['team'].append(matchup.home_team)
        match_dict['opponent'].append(matchup.away_team)
        match_dict['team_points'].append(matchup.home_points)
        match_dict['opponent_points'].append(matchup.away_points)
        match_dict['week'].append(matchup.week)
        match_dict['season'].append(matchup.season)

        match_dict['team'].append(matchup.away_team)
        match_dict['opponent'].append(matchup.home_team)
        match_dict['team_points'].append(matchup.away_points)
        match_dict['opponent_points'].append(matchup.home_points)
        match_dict['week'].append(matchup.week)
        match_dict['season'].append(matchup.season)

    # Make Ranking DF
    rankings = rankings_api_instance.get_rankings(year=year)

    for week in rankings:
        for school in week.polls[0].ranks:
            rank['team'].append(school.school)
            rank['rank'].append(school.rank)
            rank['week'].append(week.week)
            rank['season'].append(week.season)

    # Advanced Statistics DF
    advanced_stats = advanced_stats_api_instance.get_advanced_team_season_stats(year=year)

    for stat in advanced_stats:
        advanced_dict['team'].append(stat.team)
        advanced_dict['season'].append(stat.season)

        advanced_dict['off_total_ppa'].append(stat.offense.total_ppa)
        advanced_dict['off_success_rate'].append(stat.offense.success_rate)

        advanced_dict['def_total_ppa'].append(stat.defense.total_ppa)
        advanced_dict['def_success_rate'].append(stat.defense.success_rate)

        try:
            advanced_dict['def_total_yards'].append(stat.defense.line_yards_total + stat.defense.open_field_yards_total)
        except:
            if stat.defense.open_field_yards_total is None:
                advanced_dict['def_total_yards'].append(stat.defense.line_yards_total)
            elif stat.defense.line_yards_total is None:
                advanced_dict['def_total_yards'].append(stat.defense.open_field_yards_total)

    # Statistics DF
    stats = stats_api_instance.get_team_season_stats(year=year)

    stats_dict_temp = {'team': []}
    for team in teams:
        if team.classification == 'fbs':
            stats_dict_temp['team'].append(team.school)

        stats_dict_temp['season'] = [None] * len(stats_dict_temp['team'])
        stats_dict_temp['games'] = [None] * len(stats_dict_temp['team'])
        stats_dict_temp['totalYards'] = [None] * len(stats_dict_temp['team'])
        stats_dict_temp['netPassingYards'] = [None] * len(stats_dict_temp['team'])
        stats_dict_temp['rushingYards'] = [None] * len(stats_dict_temp['team'])
        stats_dict_temp['passingTDs'] = [None] * len(stats_dict_temp['team'])
        stats_dict_temp['rushingTDs'] = [None] * len(stats_dict_temp['team'])
        stats_dict_temp['possessionTime'] = [None] * len(stats_dict_temp['team'])
        stats_dict_temp['kickReturnTDs'] = [None] * len(stats_dict_temp['team'])
        stats_dict_temp['puntReturnTDs'] = [None] * len(stats_dict_temp['team'])
        stats_dict_temp['turnovers'] = [None] * len(stats_dict_temp['team'])
        stats_dict_temp['penaltyYards'] = [None] * len(stats_dict_temp['team'])
        stats_dict_temp['sacks'] = [None] * len(stats_dict_temp['team'])
        stats_dict_temp['tacklesForLoss'] = [None] * len(stats_dict_temp['team'])
        stats_dict_temp['passesIntercepted'] = [None] * len(stats_dict_temp['team'])

    for num in stats:
        if list(stats_dict_temp.keys()).count(num.stat_name) == 1:
            try:
                stats_dict_temp[num.stat_name][stats_dict_temp['team'].index(num.team)] = num.stat_value
                stats_dict_temp['season'][stats_dict_temp['team'].index(num.team)] = num.season
            except:
                continue
    
    if 'stats_dict' in locals():
        for key in stats_dict_temp:
            for item in stats_dict_temp[key]:
                stats_dict[key].append(item)
    else:
        stats_dict = stats_dict_temp

matchup_df = pd.DataFrame(data=match_dict)
matchup_df['win'] = np.where(matchup_df['team_points'] > matchup_df['opponent_points'], 1, 0)
ranking_df = pd.DataFrame(data=rank)
advanced_df = pd.DataFrame(data=advanced_dict)
stats_df = pd.DataFrame(data=stats_dict)

# Combine DFs into One
matchup_ranking_df = matchup_df.merge(ranking_df, how='left', on=['team','week','season'])
matchup_ranking_df['rank'].fillna(26, inplace = True)

# Get Opponent Rank
matchup_ranking_df['opponent_rank'] = [None] * len(matchup_ranking_df)

for i in range(len(matchup_ranking_df)):
    team = matchup_ranking_df['opponent'][i]
    week = matchup_ranking_df['week'][i]
    season = matchup_ranking_df['season'][i]

    if len(ranking_df[(ranking_df['team'] == str(team)) & (ranking_df['week'] == week) & (ranking_df['season'] == season)]['rank']) == 0:
        matchup_ranking_df['opponent_rank'][i] = 26
    else:
        opp_rank = ranking_df[(ranking_df['team'] == str(team)) & (ranking_df['week'] == week)]['rank'].reset_index()
        matchup_ranking_df['opponent_rank'][i] = opp_rank['rank'][0]

mr_advstats_df = matchup_ranking_df.merge(advanced_df, how='left', on=['team','season'])
mr_advstats_df.dropna(axis=0, inplace = True)
mr_advstats_df = mr_advstats_df.reset_index(drop=True)

full_df = mr_advstats_df.merge(stats_df, how = 'left', on=['team', 'season'])
full_df.rename(columns={'turnovers':'off_turnovers'}, inplace = True)

#full_df.dropna(axis = 0, inplace = True)
full_df = full_df.reset_index(drop=True)

# Define Per Game Metrics
full_df['def_per_game_ppa'] = full_df['def_total_ppa']/full_df['games']
full_df['off_per_game_ppa'] = full_df['off_total_ppa']/full_df['games']
full_df['total_yards_per_game'] = full_df['totalYards']/full_df['games']
full_df['passing_yards_per_game'] = full_df['netPassingYards']/full_df['games']
full_df['rushing_yards_per_game'] = full_df['rushingYards']/full_df['games']
full_df['rushing_yards_per_game'] = full_df['rushingYards']/full_df['games']
full_df['TDs_per_game'] = (full_df['passingTDs']+full_df['rushingTDs']+full_df['kickReturnTDs']+full_df['puntReturnTDs'])/full_df['games']
full_df['time_of_possession_per_game'] = full_df['possessionTime']/full_df['games']
full_df['off_turnovers_per_game'] = full_df['off_turnovers']/full_df['games']
full_df['penalty_yards_per_game'] = full_df['penaltyYards']/full_df['games']
full_df['sacks_per_game'] = full_df['sacks']/full_df['games']
full_df['tackles_for_loss_per_game'] = full_df['tacklesForLoss']/full_df['games']
full_df['def_interceptions_per_game'] = full_df['passesIntercepted']/full_df['games']

# Make Teams into Number Codes and Normalize Data
norm_df = full_df[['team','opponent', 'team_points', 'opponent_points', 'season', 'week', 'rank', 'opponent_rank', 'win', 'off_success_rate','def_success_rate', 'def_per_game_ppa',
                   'off_per_game_ppa', 'total_yards_per_game', 'passing_yards_per_game', 'rushing_yards_per_game', 'TDs_per_game', 'time_of_possession_per_game',
                   'off_turnovers_per_game', 'penalty_yards_per_game', 'sacks_per_game', 'tackles_for_loss_per_game', 'def_interceptions_per_game']]

norm_df["team_code"] = norm_df["team"].astype("category").cat.codes
norm_df["opp_code"] = norm_df["opponent"].astype("category").cat.codes

norm_df['rank'] = norm_df['rank']/ norm_df['rank'].max()
norm_df['opponent_rank'] = norm_df['opponent_rank']/ norm_df['opponent_rank'].max()
norm_df['def_per_game_ppa'] = norm_df['def_per_game_ppa']/ norm_df['def_per_game_ppa'].max()
norm_df['off_per_game_ppa'] = norm_df['off_per_game_ppa']/ norm_df['off_per_game_ppa'].max()
norm_df['total_yards_per_game'] = norm_df['total_yards_per_game']/ norm_df['total_yards_per_game'].max()
norm_df['passing_yards_per_game'] = norm_df['passing_yards_per_game']/ norm_df['passing_yards_per_game'].max()
norm_df['rushing_yards_per_game'] = norm_df['rushing_yards_per_game']/ norm_df['rushing_yards_per_game'].max()
norm_df['TDs_per_game'] = norm_df['TDs_per_game']/ norm_df['TDs_per_game'].max()
norm_df['time_of_possession_per_game'] = norm_df['time_of_possession_per_game']/ norm_df['time_of_possession_per_game'].max()
norm_df['off_turnovers_per_game'] = norm_df['off_turnovers_per_game']/ norm_df['off_turnovers_per_game'].max()
norm_df['penalty_yards_per_game'] = norm_df['penalty_yards_per_game']/ norm_df['penalty_yards_per_game'].max()
norm_df['sacks_per_game'] = norm_df['sacks_per_game']/ norm_df['sacks_per_game'].max()
norm_df['tackles_for_loss_per_game'] = norm_df['tackles_for_loss_per_game']/ norm_df['tackles_for_loss_per_game'].max()
norm_df['def_interceptions_per_game'] = norm_df['def_interceptions_per_game']/ norm_df['def_interceptions_per_game'].max()

# Save DFs to Excel
full_df.to_excel(f"{full_path}/{full_file}")
norm_df.to_excel(f"{norm_path}/{norm_file}")
