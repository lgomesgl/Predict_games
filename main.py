from Data.data import main_data
from Models.Predict_points import preprocessing, best_model
from Statistic.static import minimum_odd, kelly_criterion

FILE_NAME = 'Dataset/Results_season_22_23_NP.csv'
data = main_data(FILE_NAME, point_at_odd=215)
df_x, df_y, x_train, x_test, y_train, y_test = preprocessing(data)
score = best_model(df_x, df_y, x_train, x_test, y_train, y_test)

odd_min = minimum_odd(score)
odd = 1.5
if odd > odd_min:
    print(kelly_criterion(score, odd))