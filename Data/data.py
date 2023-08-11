import pandas as pd
import numpy as np

'''
    Dataset Web:
    https://www.basketball-reference.com/leagues/NBA_2022_games.html
'''
def raw_data(file_name):
    return pd.read_csv(file_name)

def rename_and_crop_some_columns(data):
    data = data.rename(columns={'Start (ET)':'Time', 'Visitor/Neutral':'Away', 'PTS':'Away_points', 'Home/Neutral':'Home','PTS.1':'Home_points', 'Unnamed: 6': 'Box score', 'Unnamed: 7': 'OT'})
    data = data.drop(['Box score','Attend.','Notes'], axis=1)
    return data

def clean_data(data, ot):
    day_and_month = data['Date'].apply(lambda x: x.split(' '))    
    day = [i[0] for i in day_and_month]
    month = [i[1] for i in day_and_month]
    data['Day'] = day
    data['Month'] = month
    data = data.drop(['Date'], axis=1)

    data['Time'] = data['Time'].apply(lambda x: x[:-1])
    
    if not ot:
        data['OT'] = data['OT'].isnull()
        data = data[(data['OT'] == True)]
        data = data.drop(['OT'], axis=1)
        
    data['Points'] = data['Away_points'] + data['Home_points']

    data = data[['Day','Month','Time','Arena','Home','Home_points','Away','Away_points','Points']]  
    return data

def change_to_classification_data(data, point_at_odd):
    def classify_the_point(points, point_at_odd):
        if points >= point_at_odd:
            return 1
        return 0
    
    data['Points'] = data['Points'].apply(lambda x: classify_the_point(x, point_at_odd))
    
    return data
        
FILE_NAME = 'Results_season_22_23_NP.csv'
def main_data():
    data_results = raw_data(FILE_NAME)
    data_results = rename_and_crop_some_columns(data_results)
    data_results = clean_data(data_results, ot=False)
    # data_results = change_to_classification_data(data_results, point_at_odd=225)

    return data_results