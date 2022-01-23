#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:17:20 2019

@author: cany
"""

#import matplotlib.pyplot as plt

import os
from PIL import Image
import numpy as np
import logging

from nuscenes.nuscenes import NuScenes

from nuscenes.map_expansion.map_api import NuScenesMap

from dataset import nuscenes_helper


nusc = NuScenes(version='v1.0-trainval', dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', verbose=True)
scenes = nusc.scene

nusc_map_sin_onenorth = NuScenesMap(dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', map_name='singapore-onenorth')
nusc_map_sin_hollandvillage = NuScenesMap(dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', map_name='singapore-hollandvillage')
nusc_map_sin_queenstown = NuScenesMap(dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', map_name='singapore-queenstown')
nusc_map_bos = NuScenesMap(dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', map_name='boston-seaport')

all_samples = nusc.sample

target_dir = '/srv/beegfs02/scratch/tracezuerich/data/cany/deneme'

#layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']

layer_names = ['drivable_area','ped_crossing', 'walkway', 'carpark_area','road_segment','lane']
    
all_ids = np.arange(len(scenes))
logging.error('TOTAL NUMBER OF SCENES : '  + str(len(scenes)))
all_samples = []

total_slices = int(len(layer_names)+2)

label_creator_array = np.zeros((900,1600,total_slices),np.float32)

for k in range(total_slices):
    label_creator_array[...,k] = 2**k


cur_camera = 'CAM_FRONT'
logging.error('THIS IS CAMERA ' + cur_camera)

for k in range(len(all_ids)):
    
    logging.error('CURRENTLY ' + str(k) + ' of ' + str(len(all_ids)))
    
    my_scene = scenes[all_ids[k]]
    

    
    current_dir = os.path.join(target_dir,'scene'+my_scene['token'])
    
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    
    first_sample_token = my_scene['first_sample_token']
    last_sample_token = my_scene['last_sample_token']
    first_sample_ind = nusc.getind('sample',first_sample_token)
    last_sample_ind = nusc.getind('sample',last_sample_token)
    
    scene_samples = np.arange(first_sample_ind,last_sample_ind+1)
    all_samples.append(scene_samples)

    for m in range(len(scene_samples)):
        
        my_sample = nusc.sample[scene_samples[m]]
#        my_scene = nusc.get('scene',my_sample['scene_token'])
        
        log_record = nusc.get('log', my_scene['log_token'])
        log_location = log_record['location']
        
        if log_location=='singapore-onenorth':
            my_map_api = nusc_map_sin_onenorth
        elif log_location=='singapore-hollandvillage':
            my_map_api = nusc_map_sin_hollandvillage
        elif log_location=='singapore-queenstown':
            my_map_api = nusc_map_sin_queenstown
        else:
            my_map_api = nusc_map_bos
        
        sample_token = my_sample['token']
        
        image, label = nuscenes_helper.get_image_and_mask(nusc,sample_token, my_map_api,layer_names=layer_names,camera_channel=cur_camera)
        len_of_number = len(str(m))
        init_str = str(m)
        for digit in range(5-len_of_number):
            init_str = '0'+init_str
            
            
        img_png = Image.fromarray(np.uint8(image))
        img_png.save(os.path.join(current_dir, 'img'+init_str+'.png'))
        
        png_label = np.uint8(np.squeeze(np.sum(label*label_creator_array,axis=-1)))
            
        img_png = Image.fromarray(png_label)
        img_png.save(os.path.join(current_dir, 'label'+init_str+'.png'))
        
#        
#        
#        img_filename = os.path.join(current_dir, 'img'+init_str+'.npy')
#        label_filename = os.path.join(current_dir, 'label'+init_str+'.npy')
#        np.save(img_filename, np.uint8(image))
#        np.save(label_filename, np.uint8(label))
#        