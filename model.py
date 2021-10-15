import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df2 = pd.read_csv('tuned_random_for.csv')
x = df2.drop(['MIS_Status', 'Unnamed: 0'], 1)
y = df2['MIS_Status']

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3, random_state=11)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


rf = RandomForestClassifier(max_depth=8, criterion='gini')
model = rf.fit(x_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

