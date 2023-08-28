import numpy as np
import pandas as pd

def read_data(path_data):
    return pd.read_csv(path_data)

def clean_data(data):
    data = data.drop(['Rk'], axis=1)
    return data

PATH_DATA = 'D:\LUCAS\Project_nba\Database\Expanded_Standings_22_23.csv'
data = read_data(PATH_DATA)