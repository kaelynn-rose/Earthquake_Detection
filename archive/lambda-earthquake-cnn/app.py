'''
This lambda function is part of the Docker container image to run the CNN classification model in the seismic_CNN.py file in this repo anytime an image is uploaded to the specified s3 bucket. The user must specify the appropriate IMAGE_WIDTH and IMAGE_HEIGHT that works with the model.

created by Kaelynn Rose
on April 23 2021

'''

import json
import boto3
import numpy as np
import PIL.Image as Image
import tensorflow as tf

# specify dimensions of image that match the image size used to train the model
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 100

IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT)
model = tf.keras.models.load_model('model') # load the model from the model folder in the Docker container

labels= np.array(open('model/labels.txt').read().splitlines()) # load the labels of the classes
s3 = boto3.resource('s3') # connect to s3 via boto3

def lambda_handler(event, context):
  bucket_name = event['Records'][0]['s3']['bucket']['name'] # get bucket name
  key = event['Records'][0]['s3']['object']['key'] # get event key

  img = readImageFromBucket(key, bucket_name).resize(IMAGE_SHAPE) # get image from bucket and resize
  img2 = np.array(img) # convert image to numpy array
    
  new_img = []
  for i in range(0,len(img2)): # convert from RGBA to grayscale using same weights used by cv2 package
    new = (0.299*(img2[i][:,0])) + (0.587*(img2[i][:,1])) + (0.114*img2[i][:,2])
    new_img.append(new)

  img = np.array(new_img)/255.0 # normalize image
  img = img.reshape(-1,IMAGE_HEIGHT,IMAGE_WIDTH,1)
  prediction = model.predict(img) # predict class of image using model in Docker container
  predicted_class = labels[np.argmax(prediction[0], axis=-1)] # find the predicted class of the image

  print('ImageName: {0}, Prediction: {1}'.format(key, predicted_class))
  print(f'The prediction is: {prediction}')

def readImageFromBucket(key, bucket_name):
  bucket = s3.Bucket(bucket_name) # specify s3 bucket
  object = bucket.Object(key)
  response = object.get()
  return Image.open(response['Body'])
