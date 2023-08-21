from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

def preprocessing(data, data_full):
    df = data.copy()
    df = pd.get_dummies(df,columns=['Day','Month','Time','Arena','Home','Away'])

    if data_full:
        scaler = MinMaxScaler()
        df[['GB_HT','SRS_HT','PS/G_HT','PA/G_HT','PS/G_AT','PA/G_AT','GB_AT','SRS_AT']] = scaler.fit_transform(df[['GB_HT','SRS_HT','PS/G_HT','PA/G_HT','PS/G_AT','PA/G_AT','GB_AT','SRS_AT']])
        # df[['SRS_HT','SRS_AT']] = scaler.fit_transform(df[['SRS_HT','SRS_AT']])
        
    df_x = df.drop(['Home_points','Away_points','Points'], axis=1)
    df_y = df['Points']

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=10)
    
    return df_x, df_y, x_train, x_test, y_train, y_test

def best_model(df_x, df_y, x_train, x_test, y_train, y_test):
    model_1 = RandomForestClassifier()
    model_2 = XGBClassifier()
    model_3 = SVC()
    model_4 = BernoulliNB()
    best_score = 0
    models = pd.DataFrame(columns=['Models','Accuracy','AUC'])
    for model in [model_1,model_2,model_3,model_4]:
    
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        # score = accuracy_score(y_pred, y_test)
        
        score_cross = cross_val_score(model,df_x,df_y)
        # print('Model_%s :%s%% accuracy with std: %s' %(model, round(score_cross.mean()*100,2), round(score_cross.std(),2)))
        
        score_auc_cross = cross_val_score(model,df_x,df_y,scoring="roc_auc")
        # print('AUC: %s with std: %s' %(round(score_auc_cross.mean(),3), round(score_auc_cross.std(),2)))
        
        if score_cross.mean() > best_score:
            best_score = score_cross.mean()
            # joblib.dump(model, 'Best_model')
        
        row = pd.DataFrame([{'Models':str(model), 'Accuracy':'%s%% std: %s' % (round(score_cross.mean()*100,2), round(score_cross.std(),2)),
                             'AUC': '%s std: %s' % (round(score_auc_cross.mean(),3), round(score_auc_cross.std(),2))}])
        models = pd.concat([models, row], ignore_index=True)
    
    print(models)
    return best_score

