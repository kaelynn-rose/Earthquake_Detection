'''



Created by Kaelynn Rose
on 4/22/2021

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

import tensorflow as tf
from tensorflow.keras import layers
keras = tf.keras

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from datetime import datetime
from joblib import Parallel,delayed

from scipy import signal
from scipy.signal import resample,hilbert
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# get envelopes
dataset = np.load(dataset_path,allow_pickle=True)
counter = 0
envelopes = []
for i in range(0,len(dataset)):
    counter += 1
    print(f'Working on trace # {counter}')
    data = dataset[i][:,2]
    sos = signal.butter(4, (1,49.9), 'bandpass', fs=100, output='sos') # filter signal from 1-50 Hz, 4th order filter
    filtered = signal.sosfilt(sos, data)
    analytic_signal = hilbert(filtered) # apply hilbert transform to get signal envelope
    amplitude_envelope = np.abs(analytic_signal) # get only positive envelope
    env_series = pd.Series(amplitude_envelope) # convert to a series to be compatible with pd.Series rolling mean calc
    rolling_obj = env_series.rolling(200) # 2-second rolling mean (100 Hz * 2 sec = 200 samples)
    rolling_average = rolling_obj.mean()
    rolling_average_demeaned = rolling_average[199:] - np.mean(rolling_average[199:])
    rolling_average_padded = np.pad(rolling_average_demeaned,(199,0),'constant',constant_values=(list(rolling_average_demeaned)[0])) # pad with zeros to remove nans created by rolling mean
    resamp = signal.resample(rolling_average_padded, 1000) # resample signal from 6000 samples to 300
    envelopes.append(resamp)
np.save('envelopes100k1000samples.npy',envelopes)




class ClassificationLSTM:

    def __init__(self,envelopes_path,label_csv_path,target):
        self.envelopes_path = envelopes_path
        self.label_csv_path = label_csv_path
        self.target = target
        self.envelopes = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.model = []
        self.ypred = []
        self.cm = []
        self.history = []
        self.epochs = []
        
        self.envelopes = np.load(self.envelopes_path,allow_pickle=True)
        self.label_csv = pd.read_csv(self.label_csv_path)
        self.labels = self.label_csv[target]
        self.labels = np.array(self.labels.map(lambda x: 1 if x == 'earthquake_local' else 0))
        
        
    def train_test_split(self, test_size,random_state):
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.envelopes, self.labels,random_state = random_state,test_size=test_size) # train test split
        self.X_train = np.reshape(self.X_train, (np.array(self.X_train).shape[0], 1, np.array(self.X_train).shape[1]))
        self.X_test = np.reshape(self.X_test, (np.array(self.X_test).shape[0], 1, np.array(self.X_test).shape[1]))
        self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape
        
    def LSTM_fit(self,epochs,metric,batch_size):
        
        self.epochs = epochs
        # set callbacks to save model at each epoch
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f'./saved_models/LSTM_100000_dataset_classification_epochs{epochs}_{format(datetime.now().strftime("%Y%m%d%h%m%s"))}',
                save_freq='epoch')
        ]

        # model design
        model = keras.Sequential()
        model.add(keras.layers.SimpleRNN(64, input_shape=(1,self.X_train.shape[2]), return_sequences=True))
        model.add(keras.layers.LSTM(64, input_shape=(1,self.X_train.shape[2]), return_sequences=True))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.LSTM(32, return_sequences=False))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=4e-5)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy', metrics=metric)
 
        # fit and predict
        self.model = model
        self.history = model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=self.epochs,callbacks=callbacks,validation_split=0.2)
        
        # plot train/test accuracy history
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(7,7))
        ax.plot(self.history.history['accuracy'])
        ax.plot(self.history.history['val_accuracy'])
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['train','test'])
        plt.savefig('model_accuracy.png')
        plt.show()
        
                

    def LSTM_evaluate(self):
        
        self.y_pred = self.model.predict(self.X_test) # get predictions
        
        print('Evaluating model on test dataset')

        # evaluate model
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print(f'Test data accuracy: {test_acc}')

        predicted_classes = self.model.predict_classes(self.X_test)
        accuracy = accuracy_score(self.y_test,predicted_classes)
        precision = precision_score(self.y_test,predicted_classes)
        recall = recall_score(self.y_test,predicted_classes)
        print(f'The accuracy of the model is {accuracy}, the precision is {precision}, and the recall is {recall}')
        
        # save model
        saved_model_path = f'./saved_models/LSTM_classification_acc{accuracy}_prec{precision}_rec{recall}_epochs{self.epochs}_{format(datetime.now().strftime("%Y%m%d"))}' # _%H%M%S
        # Save entire model to a HDF5 file
        self.model.save(saved_model_path)
        
        # confusion matrix
        self.cm = confusion_matrix(self.y_test,predicted_classes)
        
        # plot confusion matrix
        plt.style.use('default')
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, display_labels=['not earthquake','earthquake'])
        disp.plot(cmap='Blues', values_format='')
        plt.title(f'Classification CNN Results ({self.epochs} epochs)')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()


# set data paths
dataset_path = 'partial_signal_dataset100000.npy'
envelopes_path = 'envelopes100k.npy'
label_csv_path = 'partial_signal_df100000.csv'

# use ClassificationLSTM
model_c2 = ClassificationLSTM(envelopes_path,label_csv_path,'trace_category')
model_c2.train_test_split(test_size=0.25,random_state=44)
model_c2.LSTM_fit(epochs=50,metric='accuracy',batch_size=64)
model_c2.LSTM_evaluate()




class RegressionLSTM:

    def __init__(self,envelopes_path,label_csv_path,target):
        self.envelopes_path = envelopes_path
        self.label_csv_path = label_csv_path
        self.target = target
        self.envelopes = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.model = []
        self.ypred = []
        self.cm = []
        self.history = []
        self.epochs = []
        
        self.envelopes = np.load(self.envelopes_path,allow_pickle=True)
        self.label_csv = pd.read_csv(self.label_csv_path)
        self.labels = self.label_csv[target]
        
        eq_envelopes = np.array(self.envelopes)[self.label_csv['trace_category'] == 'earthquake_local']
        print(len(eq_envelopes))
        eq_labels = self.label_csv[self.target][self.label_csv['trace_category'] == 'earthquake_local']
        print(len(eq_labels))
        
        self.envelopes = eq_envelopes
        self.labels = eq_labels
        
    def train_test_split(self, test_size,random_state):
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.envelopes, self.labels,random_state = random_state,test_size=test_size) # train test split
        self.X_train = np.reshape(self.X_train, (np.array(self.X_train).shape[0], 1, np.array(self.X_train).shape[1]))
        self.X_test = np.reshape(self.X_test, (np.array(self.X_test).shape[0], 1, np.array(self.X_test).shape[1]))
        self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape
        
    def LSTM_fit(self,epochs,batch_size):
        self.epochs = epochs
        # set callbacks to save model at each epoch
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=f'./saved_models/LSTM_100000_dataset_regression_epochs{epochs}_{format(datetime.now().strftime("%Y%m%d%h%m%s"))}',
                save_freq='epoch')
        ]

        # model design
        model = keras.Sequential()
        model.add(keras.layers.LSTM(32, input_shape=(1, 300), return_sequences=True))
        model.add(keras.layers.LSTM(32, return_sequences=False))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(1, activation='linear'))
        opt = keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=opt,
                      loss='mse')
        self.model = model
        
        # fit
        self.model = model
        self.history = self.model.fit(self.X_train, self.y_train, batch_size=batch_size,epochs=epochs,callbacks=callbacks,validation_split=0.2)
        
        # plot train/test accuracy history
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(7,7))
        ax.plot(self.history.history['loss'])
        ax.plot(self.history.history['val_loss'])
        ax.set_title('Model Loss (MSE)')
        ax.set_ylabel('MSE')
        ax.set_xlabel('Epoch')
        ax.legend(['train','test'])
        plt.savefig('model_loss.png')
        plt.show()
        
                

    def LSTM_evaluate(self):
        
        self.y_pred = self.model.predict(self.X_test) # get predictions
        
        print('Evaluating model on test dataset')

        # evaluate model
        test_loss = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print(f'Test data loss: {test_loss}')

        # save model
        saved_model_path = f'./saved_models/LSTM_regression_loss{test_loss}_epochs{self.epochs}_{format(datetime.now().strftime("%Y%m%d"))}' # _%H%M%S
        # Save entire model to a HDF5 file
        self.model.save(saved_model_path)
        
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(7,7))
        ax.scatter(self.y_test,self.y_pred,alpha=0.05)
        point1 = [0,0]
        point2 = [1200,1200]
        xvalues = [point1[0], point2[0]]
        yvalues = [point1[1], point2[1]]
        ax.plot(xvalues,yvalues,color='blue')
        ax.set_ylabel('Predicted Value',fontsize=14)
        ax.set_xlabel('Observed Value',fontsize=14)
        ax.set_title(f'Regression LSTM Results S-Wave Arrival Times | ({self.epochs} epochs)',fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlim([0,1200])
        ax.set_ylim([0,1200])
        plt.tight_layout()
        plt.savefig('true_vs_predicted.png')
        plt.show()


# set data paths
dataset_path = 'partial_signal_dataset100000.npy'
envelopes_path = 'envelopes100k.npy'
label_csv_path = 'partial_signal_df100000.csv'

# use RegressionLSTM
model_r1 = RegressionLSTM(envelopes_path,label_csv_path,'p_arrival_sample')
model_r1.train_test_split(test_size=0.25,random_state=44)
model_r1.LSTM_fit(epochs=50,batch_size=32)
model_r1.LSTM_evaluate()


# p-wave
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(model_r1.y_test,model_r1.y_pred,alpha=0.01)
point1 = [0,0]
point2 = [1200,1200]
xvalues = [point1[0], point2[0]]
yvalues = [point1[1], point2[1]]
ax.plot(xvalues,yvalues,color='blue')
ax.set_ylabel('Predicted Value',fontsize=14)
ax.set_xlabel('Observed Value',fontsize=14)
ax.set_title(f'Regression LSTM Results S-Wave Arrival Times | (50 epochs)',fontsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_xlim([200,1200])
ax.set_ylim([200,1200])
plt.tight_layout()
plt.savefig('true_vs_predicted.png')
plt.show()

# s-wave
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(model_r1.y_test,model_r1.y_pred,alpha=0.05)
point1 = [0,0]
point2 = [5000,5000]
xvalues = [point1[0], point2[0]]
yvalues = [point1[1], point2[1]]
ax.plot(xvalues,yvalues,color='blue')
ax.set_ylabel('Predicted Value',fontsize=14)
ax.set_xlabel('Observed Value',fontsize=14)
ax.set_title(f'Regression LSTM Results S-Wave Arrival Times | (50 epochs)',fontsize=14)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_xlim([0,5000])
ax.set_ylim([0,5000])
plt.tight_layout()
plt.savefig('true_vs_predicted.png')
plt.show()









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
  
  
  
envelopes = np.load(envelopes_path,allow_pickle=True)
label_csv = pd.read_csv(label_csv_path)
labels = label_csv['trace_category']
labels = np.array(labels.map(lambda x: 1 if x == 'earthquake_local' else 0))
    
X_train, X_test, y_train, y_test = train_test_split(envelopes, labels,random_state = 44,test_size=0.25) # train test split
X_train = np.reshape(X_train, (np.array(X_train).shape[0], 1, np.array(X_train).shape[1]))
X_test = np.reshape(X_test, (np.array(X_test).shape[0], 1, np.array(X_test).shape[1]))
X_train.shape, X_test.shape, y_train.shape, y_test.shape
    

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