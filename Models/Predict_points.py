from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

def preprocessing(data):
    df = data.copy()
    df = pd.get_dummies(df,columns=['Day','Month','Time','Arena','Home','Away'])
    df_x = df.drop(['Home_points','Away_points','Points'], axis=1)
    df_y = df['Points']

    # scaler = MinMaxScaler()
    # df_y = np.array(df_y)
    # df_y = df_y.reshape(-1,1)
    # df_y = scaler.fit_transform(df_y)

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=10)
    
    return df_x, df_y, x_train, x_test, y_train, y_test

def best_model(df_x, df_y, x_train, x_test, y_train, y_test):
    model_1 = DecisionTreeClassifier()
    model_2 = RandomForestClassifier()
    model_3 = XGBClassifier()

    # model_2.fit(x_train, y_train)
    # y_pred_DTC = model_2.predict(x_test)
    # score_rec_DTC = accuracy_score(y_pred_DTC, y_test)
    # print(score_rec_DTC)
    # print(confusion_matrix(y_pred_DTC,y_test))

    score = cross_val_score(model_2,df_x,df_y)
    print('%s% accuracy with std: %s' %(score.mean(), score.std()))

    # model_3.fit(x_train, y_train)
    # y_pred_XGBC = model_3.predict(x_test)
    # score_rec_XGBC = accuracy_score(y_pred_XGBC, y_test)
    # print(score_rec_XGBC)

    '''model.save'''
    return score

