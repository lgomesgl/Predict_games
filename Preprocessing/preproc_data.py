from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def under_sampling(x, y):
    rus = RandomUnderSampler(random_state=42)
    df_x, df_y = rus.fit_resample(x, y)
    return df_x, df_y

def over_sampling(x,y, variant=False):
    if not variant:
        ros = RandomOverSampler(random_state=42)
        df_x, df_y = ros.fit_resample(x, y)
        return df_x, df_y    
     
    elif variant == 'SMOTE':
        df_x, df_y = SMOTE().fit_resample(x,y)
        return df_x, df_y
    
    elif variant == 'ADASYN':
        df_x, df_y = ADASYN().fit_resample(x,y)
        return df_x, df_y       
    
def preprocessing(data, data_full, over_samp):
    df = data.copy()
    df = pd.get_dummies(df,columns=data.select_dtypes(exclude=np.number).columns.tolist())

    if data_full:
        scaler = MinMaxScaler()
        df[['GB_HT','SRS_HT','PS/G_HT','PA/G_HT','PS/G_AT','PA/G_AT','GB_AT','SRS_AT']] = scaler.fit_transform(df[['GB_HT','SRS_HT','PS/G_HT','PA/G_HT','PS/G_AT','PA/G_AT','GB_AT','SRS_AT']])
        # df[['SRS_HT','SRS_AT']] = scaler.fit_transform(df[['SRS_HT','SRS_AT']])
        
    df_x = df.drop(['Home_points','Away_points','Points'], axis=1)
    df_y = df['Points']

    if over_samp:
        df_x, df_y = over_sampling(df_x, df_y, variant=False)
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=10)
    
    return df_x, df_y, x_train, x_test, y_train, y_test