import pandas as pd
import numpy as np
from data_conference_standings import main_data_conference

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
        
def new_columns_from_conference(data_results, data_conference):
    data = data_results.copy()
    
    # Home team
    ratio_win_lose_home = []
    diff_win_home = []
    ps_g_home, pa_g_home = [], []
    srs_home = []
    for team in data_results['Home']:
        ratio_win_lose_home.append(data_conference[(data_conference['Team'] == team)]['W/L%'].values[0])
        diff_win_home.append(data_conference[(data_conference['Team']) == team]['GB'].values[0])
        ps_g_home.append(data_conference[(data_conference['Team']) == team]['PS/G'].values[0])
        pa_g_home.append(data_conference[(data_conference['Team']) == team]['PA/G'].values[0])
        srs_home.append(data_conference[(data_conference['Team']) == team]['SRS'].values[0])
        
    data['W/L%_HT'] = ratio_win_lose_home
    data['GB_HT'] = diff_win_home
    data['PS/G_HT'] = ps_g_home
    data['PA/G_HT'] = pa_g_home
    data['SRS_HT'] = srs_home
    
    # Away team
    ratio_win_lose_away = []
    diff_win_away = []
    ps_g_away, pa_g_away = [], []
    srs_away = []
    for team in data_results['Away']:
        ratio_win_lose_away.append(data_conference[(data_conference['Team'] == team)]['W/L%'].values[0])
        diff_win_away.append(data_conference[(data_conference['Team']) == team]['GB'].values[0])
        ps_g_away.append(data_conference[(data_conference['Team']) == team]['PS/G'].values[0])
        pa_g_away.append(data_conference[(data_conference['Team']) == team]['PA/G'].values[0])
        srs_away.append(data_conference[(data_conference['Team']) == team]['SRS'].values[0])
        
    data['W/L%_AT'] = ratio_win_lose_away
    data['GB_AT'] = diff_win_away
    data['PS/G_AT'] = ps_g_away
    data['PA/G_AT'] = pa_g_away
    data['SRS_AT'] = srs_away    
    
    data = data[['Day','Month','Time','Home','Home_points','Away','Away_points','W/L%_HT','GB_HT','PS/G_HT','PA/G_HT','SRS_HT','W/L%_AT','GB_AT','PS/G_AT','PA/G_AT','SRS_AT','Points']]
    # data = data[['Day','Month','Time','Arena','Home','Home_points','Away','Away_points','SRS_HT','SRS_AT','Points']]
    return data
        
def main_data(file_name, point_at_odd, data_full):
    data_results = raw_data(file_name)
    data_results = rename_and_crop_some_columns(data_results)
    data_results = clean_data(data_results, ot=False)
    data_results = change_to_classification_data(data_results, point_at_odd)
    data_conference = main_data_conference()
    data_results_conference = new_columns_from_conference(data_results, data_conference)
    
    if data_full:
        return data_results_conference
    return data_results