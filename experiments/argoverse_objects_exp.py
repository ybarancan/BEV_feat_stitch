# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:57:24 2018

@author: ybara
"""
import numpy as np

#import tensorflow as tf

experiment_name = 'argoverse_exp'
dataset='Argoverse'

log_dir = '/scratch_net/catweazle/cany/argoverse_classwise/logdir'

argo_track_path = '/srv/beegfs02/scratch/tracezuerich/data/datasets/Argoverse_v2/argoverse-tracking/all_logs'
argo_labels_path = '/srv/beegfs02/scratch/tracezuerich/data/cany/argoverse'

original_image_size = (1920,1200)


num_frames=3

min_ratio=16
# Training settings
batch_size = 1
learning_rate = 0.000002


num_total_classes = 8
num_classes=1
num_bev_classes = 8
num_static_classes=1
num_object_classes=7

ARGOVERSE_CLASS_NAMES = [
    'drivable_area', 'vehicle', 'pedestrian', 'large_vehicle', 'bicycle', 'bus',
    'trailer', 'motorcycle',
]



bev_classes=[
    'drivable_area', 'vehicle', 'pedestrian', 'large_vehicle', 'bicycle', 'bus',
    'trailer', 'motorcycle',
]




bev_positive_weights = np.array([1.2,5.2,40.0,20.0,80.0,30.0,40.0,80.0,0.1])
bev_negative_weights = np.ones_like(bev_positive_weights)

image_object_positive_weights =  np.array([5.2,15.0,20.0,50.0,40.0,50.0,60.0,0.1])

transformer_d = 128
#names_of_classes= ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
layer_names = ['drivable_area']

train_eval_frequency = 2000
val_eval_frequency = 2000

use_occlusion=True

map_extents = [-25,1,25,50]
use_grid=False

extents = [-25,25,1,50]
resolution = 0.25
map_resolution = 0.25
extra_space = [96,100]

project_extents = [-25,25,1,50]
project_resolution = 0.25
project_extra_space = [96,100]
project_patch_size = (int((project_extents[1] - project_extents[0])/project_resolution + project_extra_space[0]),int((project_extents[3] - project_extents[2])/project_resolution + project_extra_space[1]))

project_base_patch_size = (int((project_extents[1] - project_extents[0])/project_resolution),int((project_extents[3] - project_extents[2])/project_resolution ))


downsample_ratio = 1

feature_downsample=4

total_image_size = (int((extents[1] - extents[0])/resolution + extra_space[0]),int((extents[3] - extents[2])/resolution + extra_space[1]))
patch_size=total_image_size

label_patch_size= (int((extents[1] - extents[0])/resolution ),int((extents[3] - extents[2])/resolution ))

camera_image_patch_size = (400,640)
        
use_pretrained_resnet=False

bev_downsample_ratio = 4
