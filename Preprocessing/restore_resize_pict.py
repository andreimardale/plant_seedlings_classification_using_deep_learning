#!/bin/bash
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from shutil import copyfile



if __name__== "__main__":                
    path_train = './data/train/'  
    path_non_square = './data/non_square/'  
    path_resize = './data/resized/'     
    print(os.getcwd())
    
    class_plants = os.listdir(path_non_square)
    for cd in class_plants:  
        class_list_plants = os.listdir(path_non_square + cd)
        for img_plant in class_list_plants:    
            print(path_non_square+cd+'/'+img_plant)
            #print(path_train+cd+'/'+img_plant)
            copyfile(path_non_square+cd+'/'+img_plant, path_train+cd+'/'+img_plant)
