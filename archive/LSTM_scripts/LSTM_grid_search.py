'''
This script is an example of running a grid search using sklearn's GridSearchCV on the LSTM classification model (found in seismic_LSTM.py in this repo).

Created by Kaelynn Rose
on 4/22/2021

'''

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
keras = tf.keras

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score


# set data paths
dataset_path = 'partial_signal_dataset100000.npy'
envelopes_path = 'envelopes100k.npy'
label_csv_path = 'partial_signal_df100000.csv'


# grid search

def algorithm_pipeline(X_train_data, X_test_data, y_train_data, y_test_data,
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       do_probabilities = False):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-2,
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(X_train_data, y_train_data)
    
    if do_probabilities:
      pred = fitted_model.predict_proba(X_test_data)
    else:
      pred = fitted_model.predict(X_test_data)
    
    return fitted_model, pred
  
  
# load signal envelopes and corresponding labels
envelopes = np.load(envelopes_path,allow_pickle=True)
label_csv = pd.read_csv(label_csv_path)
labels = label_csv['trace_category']
labels = np.array(labels.map(lambda x: 1 if x == 'earthquake_local' else 0)) # set labels so that earthquakes are labeled with a 1 and noise is labeled with a 0

# train test split
X_train, X_test, y_train, y_test = train_test_split(envelopes, labels,random_state = 44,test_size=0.25) # train test split
X_train = np.reshape(X_train, (np.array(X_train).shape[0], 1, np.array(X_train).shape[1]))
X_test = np.reshape(X_test, (np.array(X_test).shape[0], 1, np.array(X_test).shape[1]))
X_train.shape, X_test.shape, y_train.shape, y_test.shape
 
 
# function to build model with param_grid parameters
def build_LSTM(epochs=10,dropout_rate=0.2,optimizer='Adam',batch_size=32,activation='relu'):

    # model design
    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(64, input_shape=(1,X_train.shape[2]), return_sequences=True))
    model.add(keras.layers.LSTM(64, input_shape=(1,X_train.shape[2]), return_sequences=True))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.LSTM(32, return_sequences=False))
    model.add(keras.layers.Dense(16, activation=activation))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics='accuracy')

    return model

# set parameter grid to loop through all possible combinations of parameters
param_grid = {'epochs':[10,50],
              'dropout_rate':[0.2,0.5],
              'optimizer':['Adam','Rmsprop'],
              'batch_size':[8,16,32,64],
              'activation':['relu','sigmoid','tanh']
              }

# fit and predict
model = KerasClassifier(build_fn = build_LSTM, verbose=1)
model, pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model, param_grid, cv=5, scoring_fit='neg_log_loss')

print(model.best_score_)
print(model.best_params_)
