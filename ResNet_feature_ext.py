#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:55:01 2020

@author: ubuntu
"""

from keras.applications.resnet import ResNet50
from keras.models import Model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

model = ResNet50(weights='imagenet', include_top=True)
model2 = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
#model2.summary()

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory=r"./Dataset/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

predict = model2.predict_generator(test_generator, 15851, verbose=1)

# ----------
# create df
path_list = np.array(test_generator.filenames)
path_list = path_list[:,np.newaxis]
name_list = np.array([path_list[i][0].split('/')[1].split('_')[1][:-4] for i in range(15851)])
name_list = name_list[:,np.newaxis]

name=['path','label']+['w'+str(i) for i in range(2048)]
test=pd.DataFrame(columns=name, data=np.hstack((path_list,name_list,predict)))
print(test)
test.to_csv('feature_ResNet.csv')

