'''
This script contains two LSTM model classes, which can be used to run regression or classification RNN models. See below each class for examples of how to use each class.

This script:

    1. Calculates signal envelopes from raw signal data
    2. Contains the ClassificationLSTM Class to perform classification using an LSTM RNN model on signal data, to classify signals as 'earthquakes' or 'noise'. Outputs:
            * Confusion matrix and plot
            * Accuracy history plot
            * Test accuracy values
            
    3. Contains the RegressionLSTM Class to perform regression using an LSTM regression model on signal data. The model can be used to predict several different target labels for earthquake signals, including earthquake magnitude, earthquake p-wave arrival time, and earthquake s-wave arrival time.
            * Model loss (MSE) values
            * Loss (MSE) history plot
            * Observed vs. predicted output scatterplot

This script requires that the user has already created signal traces using the get_signal_traces.py file in this repo. If needed, use the LSTM_grid_search.py file contained in this repo to run a grid search for best parameters for the model.

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


# set data paths
dataset_path = 'partial_signal_dataset100000.npy'
envelopes_path = 'envelopes100k.npy'
label_csv_path = 'partial_signal_df100000.csv'

# get signal envelopes for every signal
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
    resamp = signal.resample(rolling_average_padded, 300) # resample signal from 6000 samples to 300
    envelopes.append(resamp)
np.save('envelopes100k.npy',envelopes)



# LSTM Classification Class

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
        self.accuracy = []
        self.precision = []
        self.recall = []
        
        self.envelopes = np.load(self.envelopes_path,allow_pickle=True) # load envelopes from file
        self.label_csv = pd.read_csv(self.label_csv_path)
        self.labels = self.label_csv[target]
        self.labels = np.array(self.labels.map(lambda x: 1 if x == 'earthquake_local' else 0))
        
        
    def train_test_split(self, test_size,random_state):
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.envelopes, self.labels,random_state = random_state,test_size=test_size) # train test split
        self.X_train = np.reshape(self.X_train, (np.array(self.X_train).shape[0], 1, np.array(self.X_train).shape[1])) # reshape
        self.X_test = np.reshape(self.X_test, (np.array(self.X_test).shape[0], 1, np.array(self.X_test).shape[1])) # reshape
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

        predicted_classes = self.model.predict_classes(self.X_test) # predicted class for each image
        self.accuracy = accuracy_score(self.y_test,predicted_classes) # accuracy of model
        self.precision = precision_score(self.y_test,predicted_classes) # precision of model
        self.recall = recall_score(self.y_test,predicted_classes) # recall of model
        print(f'The accuracy of the model is {self.accuracy}, the precision is {self.precision}, and the recall is {self.recall}')
        
        # save model
        saved_model_path = f'./saved_models/LSTM_classification_acc{self.accuracy}_prec{self.precision}_rec{self.recall}_epochs{self.epochs}_{format(datetime.now().strftime("%Y%m%d"))}' # _%H%M%S
        # Save entire model to a HDF5 file
        self.model.save(saved_model_path)
        
        # confusion matrix
        self.cm = confusion_matrix(self.y_test,predicted_classes)
        
        # plot confusion matrix
        plt.style.use('default')
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, display_labels=['not earthquake','earthquake'])
        disp.plot(cmap='Blues', values_format='')
        plt.title(f'Classification LSTM Results ({self.epochs} epochs)')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()

# use ClassificationLSTM
model_c2 = ClassificationLSTM(envelopes_path,label_csv_path,'trace_category') # initialize class
model_c2.train_test_split(test_size=0.25,random_state=44) # train test split
model_c2.LSTM_fit(epochs=50,metric='accuracy',batch_size=64) # fit model
model_c2.LSTM_evaluate() # evaluate model



# LSTM Regression Class

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
        self.test_loss = []
        
        self.envelopes = np.load(self.envelopes_path,allow_pickle=True) # load signal envelopes from file
        self.label_csv = pd.read_csv(self.label_csv_path) # load labels from csv that correspond to enveloeps
        self.labels = self.label_csv[target] # find labels for specified target
        
        eq_envelopes = np.array(self.envelopes)[self.label_csv['trace_category'] == 'earthquake_local'] # only find signals corresponding to earthquakes, not noise
        print(len(eq_envelopes))
        eq_labels = self.label_csv[self.target][self.label_csv['trace_category'] == 'earthquake_local'] # only find labels corresponding to earthquakes, not noise
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
        self.test_loss = self.model.evaluate(self.X_test, self.y_test, verbose=1) # get MSE
        print(f'Test data loss: {self.test_loss}')

        # save model
        saved_model_path = f'./saved_models/LSTM_regression_loss{self.test_loss}_epochs{self.epochs}_{format(datetime.now().strftime("%Y%m%d"))}' # _%H%M%S
        # Save entire model to a HDF5 file
        self.model.save(saved_model_path)
        
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(7,7))
        ax.scatter(self.y_test,self.y_pred,alpha=0.01)
        point1 = [0,0]
        point2 = [1200,1200]
        xvalues = [point1[0], point2[0]]
        yvalues = [point1[1], point2[1]]
        ax.plot(xvalues,yvalues,color='blue')
        ax.set_ylabel('Predicted Value',fontsize=14)
        ax.set_xlabel('Observed Value',fontsize=14)
        ax.set_title(f'Regression LSTM Results | ({self.epochs} epochs)',fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlim([0,1200])
        ax.set_ylim([0,1200])
        plt.tight_layout()
        plt.savefig('true_vs_predicted.png')
        plt.show()


# use RegressionLSTM
model_r1 = RegressionLSTM(envelopes_path,label_csv_path,'p_arrival_sample') # initialize regression LSTM model
model_r1.train_test_split(test_size=0.25,random_state=44) # train test split
model_r1.LSTM_fit(epochs=50,batch_size=32) # fit model
model_r1.LSTM_evaluate() # get model mse



# nicer plots if needed
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
