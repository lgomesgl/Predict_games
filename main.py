import sys
import os 
sys.path.append(os.path.abspath(os.path.join('.','Data')))
from Data.data import main_data
from Preprocessing.preproc_data import preprocessing
from Models.sklearn_models import best_model
from Statistic.odds import minimum_odd, kelly_criterion
from Statistic.data import proportion_data

FILE_NAME = 'Database/Results_season_22_23_NP.csv'
data = main_data(FILE_NAME, point_at_odd=215, data_full=True)
print(proportion_data(data))
df_x, df_y, x_train, x_test, y_train, y_test = preprocessing(data, data_full=True)
best_score, models = best_model(df_x, df_y, x_train, x_test, y_train, y_test)
print(models)

odd_min = minimum_odd(best_score)
print(odd_min)
odd = 1.5
if odd > odd_min:
    print(kelly_criterion(best_score, odd))
    