# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:57:24 2018

@author: ybara
"""
import numpy as np

#import tensorflow as tf

experiment_name = 'nuscenes_exp'
dataset='Nuscenes'

load_path = '/scratch_net/catweazle/cany/mapmaker_bev_object/logdir/deeplabTrue/checkpoints/save/routine-39999'

nuscenes_root = '/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes'
nuscenes_processed_root = '/srv/beegfs02/scratch/tracezuerich/data/cany'
nuscenes_bev_root = '/srv/beegfs02/scratch/tracezuerich/data/cany/monomaps_labels_vanilla'
# Data settings
data_mode = '2D'  # 2D or 3D
#image_size = (720,1280)


num_heads=4
features_per_head=128
num_modules=3

original_image_size = (1600,900)

n_queries = 30

num_frames=5

min_ratio=16
# Training settings
batch_size = 1
learning_rate = 0.000002

num_total_classes = 6
num_classes=6
num_bev_classes = 14
num_static_classes=4
num_object_classes=10

bev_classes=[
    'drivable_area', 'ped_crossing', 'walkway', 'carpark', 'car', 'truck', 
    'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 
    'bicycle', 'traffic_cone', 'barrier'
]

bev_positive_weights = 2*np.array([0.5,3.0,2.0,3.0,4.0,5.0,5.0,3.0,10.0,48.0,20.0,24.0,24.0,10.0])
bev_negative_weights = np.ones_like(bev_positive_weights)

image_object_positive_weights = np.array([4.0,5.0,5.0,3.0,10.0,48.0,20.0,24.0,24.0,10.0,1])

transformer_d = 128
#names_of_classes= ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
layer_names = ['drivable_area','ped_crossing', 'walkway', 'carpark_area','road_segment','lane']

augment_batch = True
do_rotations = True
do_scaleaug = True
do_fliplr = True
max_tries = 2
train_eval_frequency = 2000
val_eval_frequency = 10000


FOV_range = [0.3,1]

use_occlusion=True


use_grid=False

map_extents = [-25,1,25,50]
extents = [-25,25,1,50]
resolution = 0.25
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

camera_image_patch_size = (448,800)
        
use_pretrained_resnet=False

bev_downsample_ratio = 4

n_clusters = 128
features_per_cluster = 32
attention_features = 32