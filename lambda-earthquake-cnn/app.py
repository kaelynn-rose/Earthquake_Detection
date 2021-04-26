import json
import boto3
import numpy as np
import PIL.Image as Image

import tensorflow as tf

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 100

IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT)
model = tf.keras.models.load_model('model')

labels= np.array(open('model/labels.txt').read().splitlines())
s3 = boto3.resource('s3')

def lambda_handler(event, context):
  bucket_name = event['Records'][0]['s3']['bucket']['name']
  key = event['Records'][0]['s3']['object']['key']

  img = readImageFromBucket(key, bucket_name).resize(IMAGE_SHAPE)
  img2 = np.array(img) # convert image to numpy array
    
  new_img = []
  for i in range(0,len(img2)): # convert from RGBA to grayscale using same weights used by cv2 package
    new = (0.299*(img2[i][:,0])) + (0.587*(img2[i][:,1])) + (0.114*img2[i][:,2])
    new_img.append(new)

  img = np.array(new_img)/255.0 # normalize
  img = img.reshape(-1,IMAGE_HEIGHT,IMAGE_WIDTH,1)
  prediction = model.predict(img)
  predicted_class = labels[np.argmax(prediction[0], axis=-1)]

  print('ImageName: {0}, Prediction: {1}'.format(key, predicted_class))
  print(f'The prediction is: {prediction}')

def readImageFromBucket(key, bucket_name):
  bucket = s3.Bucket(bucket_name)
  object = bucket.Object(key)
  response = object.get()
  return Image.open(response['Body'])
