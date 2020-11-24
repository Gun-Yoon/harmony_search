import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

data = pd.read_csv('total_data.csv')
data = data.drop(['protocol_type','service','flag'],axis=1)

X = data.drop(['class'], axis=1).to_numpy()
data['class'] = pd.factorize(data['class'])[0].astype(np.uint16)
y = data['class'].to_numpy()

#Mutual Information 수행
corr_data = data.drop(['class'], axis=1)
mic = mutual_info_classif(X,y)

high_score_features = []
for score, f_name in sorted(zip(mic, data.columns), reverse=True)[:len(corr_data.columns)]:
    print(f_name, score)
    high_score_features.append(f_name)