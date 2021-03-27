import h5py
import pandas as pd
import os
import lightgbm
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import json
import pickle

def sred(y):
    for i in range(1, int(y.shape[0]) - 1):
        if np.isnan(y[i]):
            y[i] = (y[i - 1] + y[i + 1]) / 2
    return y

def split_class(y):
    classes = np.zeros(y.shape[0])
    for i in range(y.shape[0]):
        if y[i] >= 90 and y[i] < 100:
            classes[i] = 1
        elif y[i] >= 100 and y[i] < 150:
            classes[i] = 2
        elif y[i] >= 150 and y[i] < 160:
            classes[i] = 3

    return classes

all_data = pd.DataFrame()
for fs in os.walk('bigsets'):
    for h5py_name in fs[2]:
        data = pd.DataFrame()
        h5py_path = os.path.join('bigsets', h5py_name)
        with h5py.File(h5py_path, 'r') as f:
            keys = list(f.keys())
            keys.remove('target1')
            try:
                keys.remove('target')
            except:
                print("Error: ", h5py_name)
            for key in keys:
                lol = f[key]
                lol = lol[:][:][0]
                data[key] = pd.Series(lol.reshape(1, lol.size)[0])
            data = data.dropna()
            data = data.T
            value = pd.Series(f['target1'][:data.shape[0]], index=data.index).dropna()
            try:
                value = pd.Series(f['target1'][:data.shape[0]], index=data.index).dropna()
            except ValueError:
                value = pd.Series(f['target1'][:], index=data.index).dropna()
            data['target'] = value
        all_data = all_data.append(data, ignore_index=True)

all_data = all_data.sample(frac=1) 

y = sred(all_data.target.values)

y[np.isnan(y)] = 116
y = split_class(y)
all_data.dropna(axis='columns',how='any', inplace=True)
all_data = all_data.apply(sred)
x = all_data.values

print(x.shape)
print(y.shape)
#
# Create training and validation sets
#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#
# Create the LightGBM data containers
#
train_data = lightgbm.Dataset(x_train, label=y_train)

test_data = lightgbm.Dataset(x_test, label=y_test)

#
# Train the model
#

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100)

y_res_train = model.predict(x_train)
print(y_res_train)
y_res_test = model.predict(x_test)

res_test = pd.DataFrame({'res_test':y_res_test, 'y_test':y_test})
res_train = pd.DataFrame({'res_train':y_res_train,'y_train':y_train})

df_train = res_train
df_test = res_test

y_true_train = df_train['y_train']
y_res_train = df_train['res_train']

y_true_test = df_test['y_test']
y_res_test = df_test['res_test']

def ff(ll):
    ll_1 = ll>=0.5
    ll_1 = ll_1.astype(int)
    return ll_1
    
    
y_true_train1 = ff(y_true_train) 
y_res_train1= ff(y_res_train) 
y_true_test1= ff(y_true_test) 
y_res_test1= ff(y_res_test) 

print('f1 for train = ', f1_score(y_true_train1, y_res_train1, average='weighted'))
print('f1 for test = ', f1_score(y_true_test1, y_res_test1, average='weighted'))

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)