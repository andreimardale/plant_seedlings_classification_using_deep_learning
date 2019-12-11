#!/bin/bash
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from shutil import copyfile

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from datetime import datetime


def mask_background(img1):
    light_green = (25, 50, 50)
    dark_green = (65, 255, 255)   

    #img1 = cv2.imread(path_img)
    hsv_p = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    #plt.imshow(hsv_p)  
    mask = cv2.inRange(hsv_p, light_green, dark_green)
    result = cv2.bitwise_and(hsv_p, hsv_p, mask=mask)
#     if (display== True):
#         plt.imshow(result, cmap="gray")
    return (result)    
    
    
if __name__== "__main__":   
    print(datetime.now() )
    train_datagen = ImageDataGenerator(preprocessing_function=mask_background)   
    
    train_generator = train_datagen.flow_from_directory(
            './data/train',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical')

    base_model = ResNet50(weights='imagenet')
    # add a global spatial average pooling layer
    x = base_model.output
    #x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 10 classes
    predictions = Dense(12, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    hist = model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=50)    
    print(datetime.now())
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("model_test1.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_test1.h5")
    print("Saved model to disk")            

    #model.layers.pop()
    model.summary()
    
