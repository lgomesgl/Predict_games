import numpy as np
import pandas as pd
'''
Player : Player's name Pos : Position
Age : Player's age Tm : Team
G : Games played GS : Games started MP : Minutes played per game 
FG : Field goals per game FGA : Field goal attempts per game FG% : Field goal percentage
3P : 3-point field goals per game 3PA : 3-point field goal attempts per game 3P% : 3-point field goal percentage
2P : 2-point field goals per game 2PA : 2-point field goal attempts per game
2P% : 2-point field goal percentage eFG% : Effective field goal percentage
FT : Free throws per game FTA : Free throw attempts per game FT% : Free throw percentage
ORB : Offensive rebounds per game DRB : Defensive rebounds per game TRB : Total rebounds per game
AST : Assists per game STL : Steals per game BLK : Blocks per game TOV : Turnovers per game
PF : Personal fouls per game
PTS : Points per game
'''
def read_data(data_path):
    return pd.read_csv(data_path, encoding='latin-1')

def clean_data(data):
    data = data.drop(['Rk','Pos','Player-additional'], axis=1)
    data = data.fillna(0.0)
    data['Player'] = data['Player'].str.upper()
    return data

PATH_DATA =  r'D:\LUCAS\Project_nba\Database\Nba_player_stats_per_game_22_23.csv'
data = read_data(PATH_DATA)
data = clean_data(data)