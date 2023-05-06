# -*- coding: utf-8 -*-
"""
file : test_predict.py

usage : python3 test_predict.py --testDir testdir --modelDir modelDir
output : None
saved files : dataframe having the name of the image and predicted class 

"""

import pandas as pd
import os 
import argparse

from tensorflow import keras
from tqdm import tqdm
import os


class Prediction:
  def __init__(self, testDir , modelPath):
    self.dataDir = testDir
    self.imgs = os.listdir(testDir)
    self.model = keras.models.load_model(modelPath)
    self.labels = ['water', 'noWater']

  def imgPrep(self):
    img_height = 224
    img_width = 224
    img_array = {}
    for filename in self.imgs:
      img_path = os.path.join(self.dataDir, filename)
      img = keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
      img_array[filename] = keras.preprocessing.image.img_to_array(img) / 255.0

    return img_array


  def getPrediction(self,img_array):

    model = self.model
    # print(img_array.shape)
    img_array = img_array.reshape(-1,224,224,3)
    pred = model.predict(img_array)
    pred = pred.ravel()
    pred =  (pred > 0.5).astype(int)
    # print(pred)
    # return pred
    return self.labels[pred[0]]


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--testDir',default='',type=str)
  parser.add_argument('--modelDir',default=os.path.join(os.getcwd() , 'model'),type=str)
  args = parser.parse_args()

  obj = Prediction(args.testDir, args.modelDir)

  img_array = obj.imgPrep()

  imgs = img_array.keys()
  predClass = []
  imgNames = []
  prob = []
  for key , val in tqdm(img_array.items()):
    imgNames.append(key)
    pred = obj.getPrediction(val)
    predClass.append(pred)


  df = pd.DataFrame()
  df['imgName'] = imgNames
  df['predClass'] = predClass

  df.to_csv("results.csv")

