import pandas as pd
import numpy as np

def read_data(path_data):
    return pd.read_excel(path_data)

def clean_data(data):
    data = data.drop(['W','L'], axis=1)

    def clean_gb_column(row):
        if row == '—':
            return 0.0
        return row

    data['GB'] = data['GB'].apply(lambda x: clean_gb_column(x)).astype(float)
    
    return data

PATH = 'D:\LUCAS\Project_nba\Database\Conference_Standings_22_23_NP.xlsx'
def main_data_conference():
    data = read_data(PATH)
    data = clean_data(data)
    return data

