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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from datetime import datetime
from joblib import Parallel,delayed

from scipy import signal
from scipy.signal import resample,hilbert



class ClassificationLSTM:

    def __init__(self,dataset_path,label_csv_path,target):
        self.dataset_path = dataset_path
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
        
        self.dataset = np.load(self.dataset_path,allow_pickle=True)
        self.label_csv = pd.read_csv(self.label_csv_path)
        self.labels = self.label_csv[target]
        self.labels = np.array(self.labels.map(lambda x: 1 if x == 'earthquake_local' else 0))
        
        counter = 0
        for i in range(0,len(self.dataset)):
            counter += 1
            print(f'Working on trace # {counter}')
            data = self.dataset[i][:,2]
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
            self.envelopes.append(resamp)
        
    def train_test_split(self, test_size,random_state):
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.envelopes, self.labels,random_state = random_state,test_size=test_size) # train test split
        self.X_train = np.reshape(self.X_train, (np.array(self.X_train).shape[0], 1, np.array(self.X_train).shape[1]))
        self.X_test = np.reshape(self.X_test, (np.array(self.X_test).shape[0], 1, np.array(self.X_test).shape[1]))
        self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape
        
    def LSTM_fit(self,epochs,metric,batch_size):
    
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
        model.add(keras.layers.LSTM(32, return_sequences=False))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=1e-6)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy', metrics=metric)
 
        # fit and predict
        self.model = model
        history = model.fit(self.X_train, self.y_train, batch_size=batch_size,epochs=epochs,callbacks=callbacks,validation_split=0.2)
        
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
        
        self.y_pred = model.predict(self.X_test) # get predictions
        
        print('Evaluating model on test dataset')

        # evaluate model
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print(f'Test data accuracy: {test_acc}')

        predicted_classes = self.model.predict_classes(X_test)
        accuracy = accuracy_score(y_test,predicted_classes)
        precision = precision_score(y_test,predicted_classes)
        recall = recall_score(y_test,predicted_classes)
        print(f'The accuracy of the model is {accuracy}, the precision is {precision}, and the recall is {recall}')
        
        # save model
        saved_model_path = f'./saved_models/LSTM_acc{accuracy}_prec{precision}_rec{recall}_epochs{epochs}_{format(datetime.now().strftime("%Y%m%d"))}' # _%H%M%S
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
label_csv_path = 'partial_signal_df100000.csv'

# use ClassificationLSTM
model_c1 = ClassificationLSTM(dataset_path,label_csv_path,'trace_category')
model_c1.train_test_split(test_size=0.25,random_state=44)
model_c1.LSTM_fit(epochs=10,metric='accuracy',batch_size=32)
model_c1.LSTM_evaluate()




class RegressionLSTM:

    def __init__(self,dataset_path,label_csv_path,target):
        self.dataset_path = dataset_path
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
        
        self.dataset = np.load(self.dataset_path,allow_pickle=True)
        self.label_csv = pd.read_csv(self.label_csv_path)
        self.labels = self.label_csv[target]
        self.labels = np.array(self.labels.map(lambda x: 1 if x == 'earthquake_local' else 0))
        
        counter = 0
        for i in range(0,len(self.dataset)):
            counter += 1
            print(f'Working on trace # {counter}')
            data = self.dataset[i][:,2]
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
            self.envelopes.append(resamp)
        
    def train_test_split(self, test_size,random_state):
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.envelopes, self.labels,random_state = random_state,test_size=test_size) # train test split
        self.X_train = np.reshape(self.X_train, (np.array(self.X_train).shape[0], 1, np.array(self.X_train).shape[1]))
        self.X_test = np.reshape(self.X_test, (np.array(self.X_test).shape[0], 1, np.array(self.X_test).shape[1]))
        self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape
        
    def LSTM_fit(self,epochs,metric,batch_size):
    
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
        model.add(keras.layers.LSTM(32, return_sequences=False))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=1e-6)
        model.compile(optimizer=opt,
                      loss='binary_crossentropy', metrics=metric)
 
        # fit and predict
        self.model = model
        history = model.fit(self.X_train, self.y_train, batch_size=batch_size,epochs=epochs,callbacks=callbacks,validation_split=0.2)
        
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
        
        self.y_pred = model.predict(self.X_test) # get predictions
        
        print('Evaluating model on test dataset')

        # evaluate model
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print(f'Test data accuracy: {test_acc}')

        predicted_classes = self.model.predict_classes(X_test)
        accuracy = accuracy_score(y_test,predicted_classes)
        precision = precision_score(y_test,predicted_classes)
        recall = recall_score(y_test,predicted_classes)
        print(f'The accuracy of the model is {accuracy}, the precision is {precision}, and the recall is {recall}')
        
        # save model
        saved_model_path = f'./saved_models/LSTM_acc{accuracy}_prec{precision}_rec{recall}_epochs{epochs}_{format(datetime.now().strftime("%Y%m%d"))}' # _%H%M%S
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
label_csv_path = 'partial_signal_df100000.csv'

# use ClassificationLSTM
model_c1 = ClassificationLSTM(dataset_path,label_csv_path,'trace_category')
model_c1.train_test_split(test_size=0.25,random_state=44)
model_c1.LSTM_fit(epochs=10,metric='accuracy',batch_size=32)
model_c1.LSTM_evaluate()
