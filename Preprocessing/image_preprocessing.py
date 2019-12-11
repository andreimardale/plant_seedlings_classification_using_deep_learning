#!/bin/bash
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from shutil import copyfile

def mask_background(img1, display = True):
    light_green = (25, 50, 50)
    dark_green = (65, 255, 255)   

    #img1 = cv2.imread(path_img)
    hsv_p = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    #plt.imshow(hsv_p)  
    mask = cv2.inRange(hsv_p, light_green, dark_green)
    result = cv2.bitwise_and(hsv_p, hsv_p, mask=mask)
    if (display== True):
        plt.imshow(result, cmap="gray")
    return (result)        
        
        
        
def crop_black_rows(img, img_mask, target_size, axis=0):
    print(img.shape)
    if img.shape[axis] <= target_size:
        return None, None
    
    length_2_crop = img.shape[axis] - target_size
    crop_bottom = 0
    crop_top = 0
    
    bottom = 0
    row = 0    
    while row < img_mask.shape[0]:
        if axis==0:
            r = np.unique(img_mask[row,:,:])
        elif axis==1:
            r = np.unique(img_mask[:,row,:])
        if r.any()!=[0]:
            bottom = row
            break
        row += 1
        
    top=0    
    row = img_mask.shape[axis] - 1    
        
    while row >= 0:
        if axis==0:
            r = np.unique(img_mask[row,:,:])
        elif axis==1:
            r = np.unique(img_mask[:,row,:])
        if r.any()!=[0]:
            top = img.shape[axis] - row - 1
            break
        row -= 1     
   
    if (top > bottom):
        crop_top = top - bottom
        if crop_top >= length_2_crop:
            crop_top = length_2_crop
        length_2_crop -= crop_top
    elif (top < bottom):     
        crop_bottom = bottom - top  
        if crop_bottom >= length_2_crop:
            crop_bottom = length_2_crop
        length_2_crop -= crop_bottom    
 
    if length_2_crop > 0:
        crop_bottom += int(np.floor(length_2_crop/2))        
        crop_top += int(np.ceil(length_2_crop/2))

    if axis == 0:    
        crop_img = img[crop_bottom:img.shape[0]-crop_top,:,:]
    elif axis==1:    
        crop_img = img[:,crop_bottom:img.shape[1]-crop_top,:]

    return crop_img 


def squared_picture(img, img_path=''):
    if np.abs(img.shape[0]-img.shape[1])>1:
        #copyfile(img_path, path_isolate+ img_plant)
             
        if img.shape[0] > img.shape[1]:
            img_mask = mask_background(img, display = False)
            crop_img = crop_black_rows(img,img_mask, img_mask.shape[1], axis=0)
        else:
            img_mask = mask_background(img, display = False)
            crop_img = crop_black_rows(img, img_mask,img_mask.shape[0], axis=1)
        print(crop_img.shape)    
        if img_path != '':
            cv2.imwrite(img_path, crop_img)
        return crop_img  
    else:
        return img


def scan_non_square_picture(root_path, resize_path, original_path):    
    class_plants = os.listdir(root_path)
    for cd in class_plants:
        class_list_plants = os.listdir(root_path + cd)
        for img_plant in class_list_plants:
            img_path = root_path + cd + '/'+ img_plant
            img = cv2.imread(img_path) 
            if img.shape[0] != img.shape[1]:
                if os.path.isdir(original_path+cd) == False:
                    os.makedirs(original_path+cd)               
                copyfile(img_path, original_path+cd+ '/'+ img_plant)
                print(img_path,'\t', img.shape)  
                img = squared_picture(img, '')
                if os.path.isdir(resize_path+cd) == False:
                    os.makedirs(resize_path+cd)                   
                cv2.imwrite(resize_path+cd+'/'+ img_plant, img)    
                
                
if __name__== "__main__":                
    path_train = './data/train/'  
    path_non_square = './data/non_square/'  
    path_resize = './data/resized/'     

    scan_non_square_picture(path_train,
                            path_resize,
                            path_non_square)
  
