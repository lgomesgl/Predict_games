from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

def best_model(df_x, df_y, x_train, x_test, y_train, y_test):
    model_1 = RandomForestClassifier() 
    model_2 = XGBClassifier()
    model_3 = SVC()
    # model_3 = SVC(kernel='linear', class_weight='balanced', probability=True) # penalize imbalance class
    model_4 = BernoulliNB()
    best_score = 0
    models = pd.DataFrame(columns=['Models','Accuracy','AUC'])
    for model in [model_1,model_2,model_3,model_4]:
    
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        # score = accuracy_score(y_test, y_pred)
        # print(score)
        
        # print(confusion_matrix(y_test, y_pred))
        
        score_cross = cross_val_score(model,df_x,df_y)      
        score_auc_cross = cross_val_score(model,df_x,df_y,scoring="roc_auc")
        
        # if score_cross.mean() > best_score:
        #     best_score = score_cross.mean()
        #     # joblib.dump(model, 'Best_model')
        
        row = pd.DataFrame([{'Models':str(model), 'Accuracy':'%s%% std: %s' % (round(score_cross.mean()*100,2), round(score_cross.std(),2)),
                             'AUC': '%s std: %s' % (round(score_auc_cross.mean(),3), round(score_auc_cross.std(),2))}])
        models = pd.concat([models, row], ignore_index=True)
    
    
    return best_score, models

