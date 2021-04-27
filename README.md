# Earthquake Detection with Deep Learning


## Introduction

The goal of this study is to train a convolutional neural network using over 100,000 seismic signal images, to classify signals into 'earthquake' and 'noise' categories and to predict characteristics such as earthquake magnitude, earthquake p-wave arrival time, and earthquake s-wave arrival time. Additionally, the best model was deployed using an AWS Lambda function and connected to a seismic data stream to predict signal classes in near-real time. This study has potential applications for faster earthquake detection.

#### CRISP-DM Process

Business understanding – A company or institution that performs earthquake monitoring could use these models and analysis for implementing deep learning into their monitoring algorithms, which have traditionally been based off of signal amplitude short-term-average/long-term-average (STA/LTA) calculations to flag earthquakes. These models could result in faster or more accurate detection of earthquakes.

Data understanding – The full dataset consists of over 1.2 million seismic signals from the STanford EArthquake Dataset (STEAD). This is a labeled dataset that has applications for testing many other types of machine learning on seismic signals.

Data preparation – The seismic data was used to create 100,000 seismic data images, which were used to train the models.

Modeling – Two types of models were used: CNN models and LSTM models were used to classify signals as 'earthquake' or 'noise', and predict earthquake magnitude, p-wave arrival time, and s-wave arrival time.

Evaluation – The models were evaluated using accuracy/precision/recall for the classification models, and mean-squared-error (MSE) loss for the regression models. The best models for each case had good performance on the training and test datsets.

Deployment – The best classification model (the CNN model) was containerized using Docker and deployed using an AWS Lambda function and s3 bucket. The Lambda function was connected to a live data stream for near-real time predictions.

## Data

For this study, I used the STanford EArthquake Dataset (STEAD) (available at https://github.com/smousavi05/STEAD), a dataset containing 1.2 million seismic signals and corresponding metadata. STEAD is a high-quality global seismic dataset for which each signal has been classified as either:

1) Local earthquakes (where 'local' means that the earthquakes were recorded within 350 km from the seismic station) or 
2) Seismic noise that is free of earthquake signals. 

Earthquakes and their p-wave and s-wave arrival times in the STEAD dataset were classified 70% manually and 30% by an autopicker. The dataset also contained a .csv file with metadata for each seismic signal comprising 35 features, including:
* network code
* receiver code
* station location
* earthquake source location
* p-wave arrival time
* s-wave arrival time
* source magnitude
* source-reciever distance
* back-azimuth of arrival
* earthquake category (i.e., 'earthquake' or 'noise')
* etc.

Each seismic sample has 3 data channels of seismic data in .h5py format along with the metadata. The three channels correspond to the north-south, east-west, and vertical components of the seismogram (the amount of ground displacement measured on each of these axes by the instrument). Each sample is 60 seconds long and sampled at 100 Hz, for a total of 6000 samples per signal. Since the class balance of the full STEAD data is 235,426 noise samples to 1,030,232 earthquake signals (about 18% noise and 82% earthquakes), I randomly sampled 400,000 earthquake signals from the full earthquake dataset and used all 235,426 noise samples to create a closer class balance of 37% noise to 63% earthquakes for a total dataset of 635426 samples (about half the original dataset). Of these samples, 100,000 were randomlly selected to train each model.

### Exploratory Data Analysis

Exploratory data analysis was performed on the 100,000 signals to inform modeling. An example of a single seismic waveform and spectrogram is shown below, along with a graph of its power spectral density (PSD):

![plot](./Figures/wave_spec_psd.png) 

Earthquakes in the dataset ranged from -0.36 to 7.9 magnitude with an average magnitude of 1.52, ranged from -3.46 km to 341.74 km source depth with an average of 15.42 km depth, and 0 km to 336.38 km from the receiving seismic station, with an average distance of 50.58 km.

![plot](./Figures/mags_depths_dists.png) 

The global distribution of earthquakes in this dataset is shown here:
![plot](./Figures/eq_map.png) 

The global distribution of seismic stations which detected the earthquakes in the dataset is shown here:
![plot](./Figures/station_map.png) 

### Image Creation

**CNN Classification & Earthquake Magnitude Prediction Regression Images**

To create images for training my convolutional neural network, I plotted both the waveform and spectrogram for the vertical component of each seismogram and saved these as separate 3x2 inch images, with the waveform images being 110x160 pixels and the spectrograms being 100x150 pixel images. I normalized the color axis of the spectrograms to the range of -10 to 25 decibels per Hz for consistency across all signals. The spectrograms were created using an NFFT of 256. These signals were plotted using the _plot_images.py_ file contained in this repo.

Here are examples of earthquake and noise spectrograms that were used to train the CNN classification and magnitude prediction models:
![plot](./Figures/earthquakes_vs_noise_cnn_images.png) 

**CNN P-Wave and S-Wave Prediction Regression Images**

Since the p-wave and s-wave arrival times can be quite close together at the scale of the images used previously, I created a new set of images using 6x2 inch dimensions. The waveform images were used since preliminary testing showed that the CNN models trained with the waveforms rather than the spectrograms for p-waves and s-waves showed better prediction results.

Here are examples of earthquake and noise waveforms that were used to train the CNN p-wave and s-wave prediction models:
![plot](./Figures/example_waveforms.png) 

## Classification CNN - 'Earthquake' or 'Noise' Prediction

The spectrogram images were labeled with values of 'earthquake' or 'noise'. I created and tested a classifying convolutional neural network model on a subset of 100,000 randomly chosen images from the set, using the _seismic_CNN.py_ script in this repo. The script first imports the 100,000 randomly chosen images from the directory, performs a train-test split, compiles and then fits a classification cnn model, and then evaluates and saves the model and produces evaluation figures so model performance can be inspected visually. The model uses callbacks to save the partially-trained model at the end of each epoch.

```
Baseline model: The accuracy of the baseline model for earthquake vs. noise prediction is is 0.53704, the precision is 0.6359130766298132, and the recall is 0.6322173089071383

Best model: The accuracy of the classification model for earthquake vs. noise prediction is 0.98532, the precision is 0.9907954040500222, and the recall is 0.9859759949463045

```

The best model had the following CNN structure:

```
Model: "sequential_19"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_6 (Conv2D)            (None, 100, 150, 32)      832       
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 50, 75, 32)        0         
_________________________________________________________________
dropout_24 (Dropout)         (None, 50, 75, 32)        0         
_________________________________________________________________
flatten_6 (Flatten)          (None, 120000)            0         
_________________________________________________________________
dense_54 (Dense)             (None, 64)                7680064   
_________________________________________________________________
dense_55 (Dense)             (None, 16)                1040      
_________________________________________________________________
dense_56 (Dense)             (None, 2)                 34        
=================================================================
Total params: 7,681,970
Trainable params: 7,681,970
Non-trainable params: 0
_________________________________________________________________
```

For the use case of using this model to detect earthquakes in near-real-time, we would want to have a balance between minimizing false negatives and false positives so that we could classify earthquakes correctly but also not classify every noise signal as an earthquake. For this case, the most important metric would be accuracy since it gives us the proportion of true positives and true negatives identified by the model.

Evaluating the test set produced the following confusion matrix:

![plot](./Figures/CNN_classifier_confusion_matrix.png) 


The plot below shows the model accuracy history over 50 epochs:

![plot](./Figures/CNN_classifier_accuracy_history.png) 


## Regression CNN - Earthquake Magnitude Prediction

For the regression CNN, I used 100,000 spectrogram images and the target variable of earthquake magnitude. I created and tested a regression convolutional neural network model on the 100,000 image set, using the _seismic_CNN.py_ script in this repo. The script first imports the 100,000 randomly chosen images from the directory, performs a train-test split, compiles and then fits a regression cnn model using the specified target, and then evaluates and saves the model and produces evaluation figures. The model uses callbacks to save the partially-trained model at the end of each epoch.

```
Baseline model: The baseline mse for earthquake magnitude is 0.9501049752369152 

Best model: The mse of the CNN regression for earthquake magnitude is 0.15895192325115204 

```

The best model had the following structure:
```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 100, 150, 64)      1664      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 50, 75, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 50, 75, 64)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 240000)            0         
_________________________________________________________________
dense_3 (Dense)              (None, 16)                3840016   
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 17        
=================================================================
Total params: 3,841,697
Trainable params: 3,841,697
Non-trainable params: 0
_________________________________________________________________
```

A plot comparing observed/actual earthquake magnitude values vs. the magnitude values predicted by the regression CNN model is shown here:

![plot](./Figures/CNN_regression_magnitude.png) 

The plot below shows the model MSE loss history over 20 epochs:

![plot](./Figures/CNN_regression_magnitude_history.png) 


## Regression CNN - P-Wave and S-Wave Arrival Time Prediction

To predict p-wave and s-wave arrival times, I used 100,000 waveform images and the target variables of p-wave and s-wave arrival time sample. I created and tested a regression convolutional neural network model on the 100,000 image set, using the _seismic_CNN.py_ script in this repo. The script first imports the 100,000 randomly chosen images from the directory, performs a train-test split, compiles and then fits a regression cnn model using the specified target, and then evaluates and saves the model and produces evaluation figures. The model uses callbacks to save the partially-trained model at the end of each epoch.

**P-Wave Prediction**

```
Baseline model: The baseline mse for earthquake magnitude is 0.9501049752369152 

Best model: The mse of the CNN regression for earthquake magnitude is 0.15895192325115204 

```

**S-Wave Prediction**

```
Baseline model: The baseline mse for earthquake magnitude is 0.9501049752369152 

Best model: The mse of the CNN regression for earthquake magnitude is 0.15895192325115204 

```

The best model had the following structure:
```
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 110, 309, 64)      1664      
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 55, 154, 64)       0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 55, 154, 64)       0         
_________________________________________________________________
flatten_5 (Flatten)          (None, 542080)            0         
_________________________________________________________________
dense_12 (Dense)             (None, 64)                34693184  
_________________________________________________________________
dense_13 (Dense)             (None, 16)                1040      
_________________________________________________________________
dense_14 (Dense)             (None, 1)                 17        
=================================================================
Total params: 34,695,905
Trainable params: 34,695,905
Non-trainable params: 0
_________________________________________________________________
```

A plot comparing the observed and predicted **p-wave** arrival sample times is shown here:

![plot](./Figures/CNN_regression_magnitude.png) 

The model loss history for p-wave arrival sample:

![plot](./Figures/CNN_regression_magnitude_history.png) 

A plot comparing the observed and predicted **s-wave** arrival sample times is shown here:

![plot](./Figures/CNN_regression_magnitude.png) 

The model loss history for p-wave arrival sample:

![plot](./Figures/CNN_regression_magnitude_history.png) 
