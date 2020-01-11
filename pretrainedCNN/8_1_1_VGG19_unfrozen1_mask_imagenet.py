#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import os, sys, datetime, pickle
import cv2, csv
import math
import itertools


from glob import glob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications import VGG19

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay


# In[2]:


TRAIN_DIRECTORY = "../data/train"
TEST_DIRECTORY = "../data/test"
#IMG_HEIGHT = 256
#IMG_WIDTH = 256
CATEGORIES = os.listdir(TRAIN_DIRECTORY)
num_classes = len(CATEGORIES)

FILTER = False


# In[3]:


def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


def load_image(size, segment=False, stem=''):
    print('Load pictures')
    path_to_all_files = TRAIN_DIRECTORY + '/*/*.png' 
    files = glob(path_to_all_files)
    
    trainImg = []
    trainLabel = []
    num = len(files)

    # Obtain images and resizing, obtain labels
    if '\\'in files[0]:
        split_motif = '\\'
    else:
        split_motif = '/'
     
    list_class = []
    for img in files:
        trainImg.append(cv2.resize(cv2.imread(img), (size, size)))  # Get image (with resizing)
        trainLabel.append(img.split(split_motif)[-2])  # Get image label (folder name)
        if img.split(split_motif)[-2] not in list_class:
            list_class.append(img.split(split_motif)[-2])

            
    with open(stem + '_list_class.txt', 'w') as f:
        for l in list_class:
            f.write(l+'\n')
    
    trainImg = np.asarray(trainImg)  # Train images set
    trainLabel = pd.DataFrame(trainLabel)  # Train labels set

    print('Preprocess pictures')
    clearTrainImg = []
    if segment == True:    
        for img in trainImg:
            image = sharpen_image(segment_plant(img))
            clearTrainImg.append(image)
        
        clearTrainImg = np.asarray(clearTrainImg)
        trainImg = clearTrainImg / 255  
    else:
        trainImg = trainImg / 255 

    #print(trainLabel.shape)

    le = preprocessing.LabelEncoder()
    encodeTrainLabels = le.fit_transform(trainLabel[0])
    clearTrainLabel = to_categorical(encodeTrainLabels)    

    return trainImg, clearTrainLabel


def load_image_test(size, segment=False, stem=''):
    print('Load pictures')
    path_to_all_files = TEST_DIRECTORY + '/*.png' 
    files = glob(path_to_all_files)
    
    trainImg = []
    
    
    with open(stem +'_test_file_list.csv','w') as f:
        for img_file in files:
            f.write(img_file.split('/')[-1] + '\n')    
            trainImg.append(cv2.resize(cv2.imread(img_file), (size, size)))  # Get image (with resizing)

    print('Number of pict in test ' + str(len(trainImg)))        
    trainImg = np.asarray(trainImg)  # Train images set


    print('Preprocess pictures')
    clearTrainImg = []
    if segment == True:    
        for img in trainImg:
            image = sharpen_image(segment_plant(img))
            clearTrainImg.append(image)
        
        clearTrainImg = np.asarray(clearTrainImg)
        trainImg = clearTrainImg / 255  
    else:
        trainImg = trainImg / 255 

    return trainImg


# In[8]:


if __name__== "__main__":   
    n_classes = 12


    #print(clearTrainLabel[0])
    #print(clearTrainLabel.shape)
    # datagen = ImageDataGenerator(
            # rotation_range=180,  # randomly rotate images in the range
            # zoom_range = 0.1, # Randomly zoom image 
            # width_shift_range=0.1,  # randomly shift images horizontally
            # height_shift_range=0.1,  # randomly shift images vertically 
            # horizontal_flip=True,  # randomly flip images horizontally
            # vertical_flip=True  # randomly flip images vertically
        # )  
        
        
    TEST_SIZE = [320]
        
    
    for size in TEST_SIZE:
        print('### Use Pict size ' + str(size))
        stem_id = "8_1_1_VGG19-unfrozen1-mask-imagenet-imgsize"        
        trainImg, clearTrainLabel = load_image(size, segment=True, stem = stem_id+str(size))
        X_train, X_valid, y_train, y_valid = train_test_split(trainImg, clearTrainLabel, 
                                                        test_size=0.2, random_state=33, 
                                                        stratify = clearTrainLabel)    
        #datagen = ImageDataGenerator()    
        
        datagen = ImageDataGenerator(
            rotation_range=180,  # randomly rotate images in the range
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally
            height_shift_range=0.1,  # randomly shift images vertically 
            horizontal_flip=True,  # randomly flip images horizontally
            vertical_flip=True  # randomly flip images vertically
        )         
        
        datagen.fit(X_train)   


        base_model = VGG19(weights= 'imagenet', include_top=False, input_shape= (size,size,3))

        x = base_model.output
        x = Flatten()(x)    
        x = Dense(1024, activation= 'relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)

        predictions = Dense(num_classes, activation= 'softmax')(x)
        model = Model(inputs = base_model.input, outputs = predictions)

        base_model.trainable = True
        set_trainable = False
        for layer in base_model.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False           
            

        model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()          

        # checkpoints

        filepath=stem_id+str(size)+".best_{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', 
                                     verbose=1, save_best_only=True, mode='max')
        filepath=stem_id+str(size)+".last_auto4.hdf5"
        checkpoint_all = ModelCheckpoint(filepath, monitor='val_accuracy', 
                                         verbose=1, save_best_only=False, mode='max')

        BATCH_SIZE = 16    
        start_time = datetime.datetime.now()
        hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), 
                                   epochs=20,
                                   validation_data=(X_valid, y_valid), 
                                   steps_per_epoch=(10 * X_train.shape[0]) / BATCH_SIZE,
                                   callbacks=[checkpoint, checkpoint_all])
        end_time = datetime.datetime.now()
        
        with open(stem_id+str(size)+"_hist_accuracy", "wb") as f:
            pickle.dump(hist.history['accuracy'], f, pickle.HIGHEST_PROTOCOL)  
        with open(stem_id+str(size)+"_hist_val_acc", "wb") as f:
            pickle.dump(hist.history['val_accuracy'], f, pickle.HIGHEST_PROTOCOL) 

        Y_train_pred = model.predict(X_train, batch_size=BATCH_SIZE)
        Y_classes = Y_train_pred.argmax(axis=-1)
        y_o = y_train.argmax(axis=-1)
        with open(stem_id + str(size) + '_train_ypred.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(y_o)
            wr.writerow(Y_classes)   
            
        Y_val_pred = model.predict(X_valid, batch_size=BATCH_SIZE)
        Y_classes = Y_val_pred.argmax(axis=-1)
        y_o = y_valid.argmax(axis=-1)
        with open(stem_id + str(size) + '_val_ypred.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(y_o)
            wr.writerow(Y_classes)
            
        model_json = model.to_json()
        with open(stem_id+str(size)+".json", "w") as json_file:
            json_file.write(model_json)                        
            
        plt.plot(hist.history['accuracy'])
        plt.plot(hist.history['val_accuracy'])
        plt.title('Training for ' +str(20)+ ' epochs')
        plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
        plt.show()  
        plt.savefig(stem_id+str(size))
                               
        with open(stem_id + '_runtime.csv', 'a') as f:
            f.write(stem_id + '\t' + str(size) + '\t' + str(start_time) + '\t' + str(end_time))                                   

        test_img = load_image_test(size, segment=True, stem = stem_id+str(size))  
        Y_pred = model.predict(test_img, batch_size=BATCH_SIZE)
        Y_classes = Y_pred.argmax(axis=-1)
        with open(stem_id + str(size) + '_ypred.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(Y_classes)  


# In[ ]:




