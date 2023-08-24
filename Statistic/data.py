def proportion_data(data):
    '''
        Simetric Data -> About max '55%/45%' proportion
        So: 0.55/0.45 = 1.2222... & 0.45/0.55 = 0.8181...
    '''
    pro_1 = data['Points'].value_counts(normalize=True)[1]
    pro_0 = data['Points'].value_counts(normalize=True)[0]
    print(data['Points'].value_counts(normalize=True))

    if pro_1/pro_0 > 0.818 and pro_1/pro_0 < 1.22:
        return 'Accuracy_score'
    return 'AUC'