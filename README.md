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

To create images for training my convolutional neural network, I plotted both the waveform and spectrogram for the vertical component of each seismogram and saved these as separate images, with the waveform images being 110x160 pixels and the spectrograms being 100x150 pixel images. I normalized the color axis of the spectrograms to the range of -10 to 25 decibels per Hz for consistency across all signals. The spectrograms were created using an NFFT of 256. These signals were plotted using the _plot_images.py_ file contained in this repo.

Here are examples of earthquake and noise spectrograms that were used to train the CNN models:
![plot](./Figures/earthquakes_vs_noise_cnn_images.png) 

**CNN P-Wave and S-Wave Prediction Regression Images**

## Classification CNN

The spectrogram images were labeled with values of 'earthquake' or 'noise'. I created and tested a classifying convolutional neural network model on a subset of 200,000 randomly chosen images from the set, using the "earthquake_cnn.py" script in this repo. The script first imports the 200,000 randomly chosen images from the directory, performs a train-test split, compiles and then fits a classification cnn model, and then evaluates and saves the model and produces evaluation figures so model performance can be inspected visually. The model uses callbacks to save the partially-trained model at the end of each epoch.

Baseline model: Earthquakes were the larger class, with 63.33% of the signals being earthquake signals. Using a sklearn's DummyClassifier with the 'stratified' strategy of generating predictions, this gives us a baseline precision of 0.633, a baseline accuracy of 0.534, and a baseline recall of 0.635.

The best model had the following CNN structure:

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_2 (Conv2D)            (None, 100, 150, 32)      832       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 50, 75, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 50, 75, 32)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 120000)            0         
_________________________________________________________________
dense_3 (Dense)              (None, 16)                1920016   
_________________________________________________________________
dropout_2 (Dropout)          (None, 16)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 10)                170       
=================================================================
Total params: 1,921,018
Trainable params: 1,921,018
Non-trainable params: 0
_________________________________________________________________
```

The best model had the following metrics when predicting on the test set:
* Accuracy: 0.9848
* Precision: 0.9840
* Recall: 0.9921

For the use case of using this model to detect earthquakes in near-real-time, we would want to have a balance between minimizing false negatives and false positives so that we could classify earthquakes correctly but also not classify every noise signal as an earthquake. For this case, the most important metric would be accuracy since it gives us the proportion of true positives and true negatives identified by the model.

Evaluating the test set produced the following confusion matrix:

![plot](./figures/confusion_matrix.png) 

The model predictions were then evaluated, and the best and worse performing images are shown here:

![plot](./figures/earthquakes_vs_noise.png) 

The plot below shows the model accuracy history over 15 epochs:

![plot](./figures/accuracy_history.png) 


## Regression CNN

For the regression CNN, I used 200,000 spectrogram images and the target variable of earthquake magnitude. I created and tested a regression convolutional neural network model on the 200,000 image set, using the "earthquake_cnn.py" script in this repo. The script first imports the 200,000 randomly chosen images from the directory, performs a train-test split, compiles and then fits a regression cnn model using the specified target, and then evaluates and saves the model and produces evaluation figures. The model uses callbacks to save the partially-trained model at the end of each epoch.

Baseline model: The baseline model is the mean of the source magnitudes in the input dataset, so a predicted magnitude of 1.5215, which would give us a baseline MSE of 0.9497.  

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 100, 150, 32)      832       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 50, 75, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 48, 73, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 24, 36, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 24, 36, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 55296)             0         
_________________________________________________________________
dense (Dense)                (None, 16)                884752    
_________________________________________________________________
dense_1 (Dense)              (None, 32)                544       
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33        
=================================================================
Total params: 904,657
Trainable params: 904,657
Non-trainable params: 0
_________________________________________________________________
```

The best model had an MSE of 0.1344 when predicting on the test set. 

A plot comparing observed/actual earthquake magnitude values vs. the magnitude values predicted by the regression CNN model is shown here:

![plot](./figures/regression_vals4.png) 

The plot below shows the model MSE loss history over 15 epochs:

![plot](./figures/model_loss.png) 

These loss values indicate that the model reaches its peak performance around epoch #4, so 15 epochs was not necessary and just resulted in overfitting of the training set. To improve model speed for a real-world monitoring application, this model would only need 4 epochs to reach good performance. 
