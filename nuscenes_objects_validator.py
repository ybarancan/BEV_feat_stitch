
import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
import cv2
import sys
sys.path.append('/home/cany/tensorflow/models/research')
sys.path.append('/home/cany/tensorflow/models/research/slim')
#import model_dropout as model
from background_generator import BackgroundGenerator
#import config.brats_system as sys_config
from tensorflow.python.client import timeline
import glob
import h5py
#from davis2017 import validation_script
import scipy.io as sio
import scipy
import math
import random
import PIL
from PIL import Image

from deeplab import common

#from skimage import io,transform
from numpy import linalg as LA
import mem_net
import scipy.ndimage as ndimage
#from dataset import Dataset
import json
from multiprocessing.dummy import Pool as ThreadPool 
#from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
#from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
#from pyquaternion import Quaternion
from pyquaternion import Quaternion
#import descartes
import os.path as osp
from copy import copy
import keypoint_ops
from nuscenes.nuscenes import NuScenes
#from nuscenes.nuscenes import NuScenesExplorer
from nuscenes.map_expansion.map_api import NuScenesMap

import token_splits

#from nuscenes.nuscenes import NuScenes
#from nuscenes.nuscenes import NuScenesExplorer
#from nuscenes.map_expansion.map_api import NuscenesMap as NuScenesMap
#from nuscenes.map_expansion.map_api import NuScenesMap
#from nuscenes.utils.geometry_utils import  box_in_image, BoxVisibility, transform_matrix
#from nuscenes.utils.geometry_utils import  view_points as orig_view_points
#import imgaug.augmenters as iaa
### EXPERIMENT CONFIG FILE #############################################################
# Set the config file of the experiment you want to run here:

#from experiments import test as exp_config
#from experiments import deneme_exp as exp_config
from experiments import nuscenes_objects_base_val as exp_config
#from experiments import pascal_exp as exp_config
#from experiments import coco_exp as exp_config
# from experiments import unet3D_bn_modified as exp_config
# from experiments import unet2D_bn_wxent as exp_config
# from experiments import FCN8_bn_wxent as exp_config

########################################################################################
means_image = np.array([123.68, 116.779, 103.939], dtype=np.single)

nusc = NuScenes(version='v1.0-trainval', dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', verbose=True)
scenes = nusc.scene

nusc_map_sin_onenorth = NuScenesMap(dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', map_name='singapore-onenorth')
nusc_map_sin_hollandvillage = NuScenesMap(dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', map_name='singapore-hollandvillage')
nusc_map_sin_queenstown = NuScenesMap(dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', map_name='singapore-queenstown')
nusc_map_bos = NuScenesMap(dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', map_name='boston-seaport')

global_const = 3.99303084

run_on_aws = False

total_label_slices = exp_config.num_classes + 2

loop=True

# Set SGE_GPU environment variable if we are not on the local host
#sys_config.setup_GPU_environment()
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''
NuScenes Dataset creation
'''

#nusc = NuScenes(version='v1.0-trainval', dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', verbose=True)
#scenes = nusc.scene
#
#nusc_map_sin_onenorth = NuScenesMap(dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', map_name='singapore-onenorth')
#nusc_map_sin_hollandvillage = NuScenesMap(dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', map_name='singapore-hollandvillage')
#nusc_map_sin_queenstown = NuScenesMap(dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', map_name='singapore-queenstown')
#nusc_map_bos = NuScenesMap(dataroot='/srv/beegfs02/scratch/tracezuerich/data/datasets/nuScenes', map_name='boston-seaport')
#
#all_samples = nusc.sample

'''
CROSS VAL GRID SEARCH THRESHOLDS FOR REPLICATION

my_thresh = np.expand_dims(np.expand_dims(np.array([0.5,0.5,0.45,0.3,
                                            0.5, 0.5, 0.5, 0.3,
                                            0.3, 0.6, 0.45, 0.4,
                                            0.45,0.5]),axis=0),axis=0)
'''



target_dir = '/srv/beegfs02/scratch/tracezuerich/data/cany'


exp_config.batch_size=1



do_eval_on_whole_videos = True

use_deeplab = True
starting_from_cityscapes =False
starting_from_imagenet =False

do_eval_frames=[3,4,5]

one_object_at_a_time_multi_object_segmentation = False

use_balanced_loss=True
softmax_aggregation_training = False
use_binary_loss = True

num_frames=exp_config.num_frames

reference_frame_index = 1
frame_interval = 3
n_frames_per_seq = exp_config.num_frames

n_seqs = n_frames_per_seq-num_frames+1
softmax_aggregation_testing = True

if softmax_aggregation_training:
    exp_config.batch_size = 1

single_frame_test=False
use_occlusion=exp_config.use_occlusion

BATCH_SIZE = exp_config.batch_size
"""
If inception pre-process is used, the inputs to query encoders are corrected through vgg processing in the tensorflow part.
Query encoders do not use masks, thus they can be simply propagated through the Resnet. Memory encoders need to be handled 
differently since if the image's range is 0-255 and mask is 0-1 then the mask is not effective through simple addition before
batch norm. 
If root block is included, inception_preprocess should be set to False.
"""
use_inception_preprocess = False
freeze_batch_norm_layers = True
multiply_labels=True
include_root_block=True
apply_same_transform='new'
rec_count = exp_config.max_tries
apply_intermediate_loss = False

if not single_frame_test:
    log_dir = os.path.join('/scratch_net/catweazle/cany/mapmaker_bev_object/logdir/deeplab'+str(use_deeplab)+'/frame_interval_'+str(frame_interval)+'/num_frames'+str(num_frames))
else:
    
    log_dir = os.path.join('/scratch_net/catweazle/cany/mapmaker_bev_object/logdir/deeplab'+str(use_deeplab)+'/single_frame')

train_results_path = os.path.join(log_dir,'train_results')
#log_dir = os.path.join('/raid/cany/mapmaker/logdir/', exp_config.experiment_name)
validation_res_path = os.path.join(log_dir,'val_results')

if not os.path.exists(train_results_path):
    os.makedirs(train_results_path)
    
if not os.path.exists(validation_res_path):
    os.makedirs(validation_res_path)
    

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

logging.error('EXPERIMENT : ' + str(exp_config.experiment_name))
logging.error('THIS IS ' + str(log_dir))


def decode_binary_labels(labels, nclass):
    bits = np.power(2, np.arange(nclass))
    return np.uint8((np.expand_dims(labels,axis=-1) & np.reshape(bits,(1, 1,-1))) > 0)

def list_directories(path):
     return [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]


def  get_clipped_grads(gvs):
    
    capped_gvs = []
    for grad, var in gvs:
        if grad == None:
            logging.error('VAR ' + str(var) + ' NONE GRAD')
        else:
            
    
            capped_gvs.append((tf.clip_by_value(grad, -10., 10.), var)) 
    return capped_gvs


def gradient_explorer(gvs,capped_gvs):
#    logging.error('GVS SHAPE : ' + str(len(gvs)))
    min_grad=1000000
    max_grad=-100000000
    for k in range(len(gvs)):
#        logging.error(str(gvs[k][0]))
#        logging.error(str(gvs[k][1]))
        
        for m in range(len(gvs[k][0])):
            if abs(gvs[k][0][m].any()) > 1000:
                logging.error('Gradient exploded')
                
            if abs(gvs[k][1][m].any()) > 1000:
                logging.error('Variable exploded')
            if math.isnan(gvs[k][0][m].any()):
                logging.error('NAN GRADIENT')               
            if math.isnan(gvs[k][1][m].any()):
                logging.error('NAN VARIABLE')
                
        if (np.mean(np.abs(gvs[k][0]))) < 0.00001:
            logging.error('VARIABLE ' + str(capped_gvs[k][1]))
            logging.error('Vanishing Gradient')
            logging.error(np.max(np.abs(gvs[k][0])))
    return 



def single_process(pair):
     
       
    camera_channel = 'CAM_FRONT'
    image_path, label_path, my_reference_sample, my_current_sample , is_reference_sample= pair
    
    
    img = Image.open(image_path)
    img.load()
    label = Image.open(label_path)
    label.load()
    
    image=np.array(img, dtype=np.uint8)

    label=np.array(label, dtype=np.uint8)
    
    orig_label = np.zeros((label.shape[0],label.shape[1],int(total_label_slices )))
    
    rem = np.copy(label)
#    logging.error('Rem shape ' + str(rem.shape))
    for k in range(total_label_slices ):
        
        temp_rem = rem//(2**int(total_label_slices -k-1))
#        logging.error('TEMP REM SHAPE : ' + str(temp_rem.shape))
        
        orig_label[:,:,int(total_label_slices -k-1)] = np.copy(temp_rem)
        
        rem = rem%(2**int(total_label_slices -k-1))
        
        
    cam_token_ref = my_reference_sample['data'][camera_channel]
    cam_record_ref = nusc.get('sample_data', cam_token_ref)
    
    cs_record_ref = nusc.get('calibrated_sensor', cam_record_ref['calibrated_sensor_token'])
    cam_intrinsic = np.array(cs_record_ref['camera_intrinsic'])

    poserecord_ref = nusc.get('ego_pose', cam_record_ref['ego_pose_token'])
    
    '''
    '''
    cam_token_cur = my_current_sample['data'][camera_channel]
    cam_record_cur = nusc.get('sample_data', cam_token_cur)
    
    cs_record_cur = nusc.get('calibrated_sensor', cam_record_cur['calibrated_sensor_token'])
   
    poserecord_cur = nusc.get('ego_pose', cam_record_cur['ego_pose_token'])
    
    # bev_label = Image.open( os.path.join('/srv/beegfs02/scratch/tracezuerich/data/cany/bev_labels',  
    #                            cam_record_cur['token'] + '.png'))
    
    vis_mask = np.float32(orig_label[...,exp_config.num_classes])
    vis_mask = np.stack([vis_mask,vis_mask,vis_mask],axis=-1)
    warp_trans1 = tensorflow_project_to_ground(image,np.zeros((int(exp_config.camera_image_patch_size[0]/(4*exp_config.downsample_ratio)),int(exp_config.camera_image_patch_size[1]/(4*exp_config.downsample_ratio)))),poserecord_ref, cs_record_ref,poserecord_cur,cs_record_cur, cam_intrinsic,reference_frame=is_reference_sample)
    warp_trans2 = tensorflow_project_to_ground(image,np.zeros((int(exp_config.camera_image_patch_size[0]/(8*exp_config.downsample_ratio)),int(exp_config.camera_image_patch_size[1]/(8*exp_config.downsample_ratio)))),poserecord_ref, cs_record_ref,poserecord_cur,cs_record_cur, cam_intrinsic,reference_frame=is_reference_sample)
    warp_trans3 = tensorflow_project_to_ground(image,np.zeros((int(exp_config.camera_image_patch_size[0]/(16*exp_config.downsample_ratio)),int(exp_config.camera_image_patch_size[1]/(16*exp_config.downsample_ratio)))),poserecord_ref, cs_record_ref,poserecord_cur,cs_record_cur, cam_intrinsic,reference_frame=is_reference_sample)
        
    
    
    warped_img, warped_cover, warped_label, coordinate_transform = project_to_ground(image,label,poserecord_ref, cs_record_ref,poserecord_cur,cs_record_cur, cam_intrinsic,vis_mask,reference_frame=is_reference_sample)
    
    bev_label = np.zeros((warped_label.shape[0],warped_label.shape[1],int(total_label_slices )))
    
    rem = np.copy(warped_label)
#    logging.error('Rem shape ' + str(rem.shape))
    for k in range(total_label_slices ):
        
        temp_rem = rem//(2**int(total_label_slices -k-1))
#        logging.error('TEMP REM SHAPE : ' + str(temp_rem.shape))
        
        bev_label[:,:,int(total_label_slices -k-1)] = np.copy(temp_rem)
        
        rem = rem%(2**int(total_label_slices -k-1))
    
    
    
    new_sizes = (exp_config.camera_image_patch_size[1],exp_config.camera_image_patch_size[0])
    cropped_label = cv2.resize(label,(int(exp_config.camera_image_patch_size[1]/4),int(exp_config.camera_image_patch_size[0]/4)), interpolation = cv2.INTER_NEAREST)
    cropped_img = cv2.resize(image,new_sizes, interpolation = cv2.INTER_LINEAR)
    
    temp_label = np.zeros((cropped_label.shape[0],cropped_label.shape[1],int(total_label_slices )))
    
    rem = np.copy(cropped_label)
#    logging.error('Rem shape ' + str(rem.shape))
    for k in range(total_label_slices ):
        
        temp_rem = rem//(2**int(total_label_slices -k-1))
#        logging.error('TEMP REM SHAPE : ' + str(temp_rem.shape))
        
        temp_label[:,:,int(total_label_slices -k-1)] = np.copy(temp_rem)
        
        rem = rem%(2**int(total_label_slices -k-1))
    
    
    



    if not use_deeplab:
#        pre_img = inception_preprocess(cropped_img)
#        
        pre_warped_img = inception_preprocess(warped_img)
        pre_img=cropped_img - means_image 
#        pre_warped_img=warped_img - means_image 

    else:
        pre_img=cropped_img  
        pre_warped_img= inception_preprocess(warped_img)

#    logging.error('Pre img shape ' + str(pre_img.shape))
  
    return (pre_img, np.float32(temp_label),pre_warped_img, np.float32(bev_label),warped_cover,coordinate_transform,np.reshape(warp_trans1,[-1])[0:8],np.reshape(warp_trans2,[-1])[0:8],np.reshape(warp_trans3,[-1])[0:8])


def binary_to_png_convert(label):
    
    label = np.squeeze(label)
    label_creator_array = np.ones((label.shape[0],label.shape[1],int(exp_config.num_classes)),np.float32)
    
    for k in range(int(exp_config.num_classes)):
        label_creator_array[...,k] = 2**(k+1)
        
    png_label = np.uint8(np.squeeze(np.sum(label*label_creator_array,axis=-1)))
    return png_label


def png_to_binary(cropped_label):
    temp_label = np.ones((cropped_label.shape[0],cropped_label.shape[1],int(total_label_slices )))
    
    rem = np.copy(cropped_label)
    for k in range(total_label_slices ):
        temp_rem = rem//(2**int(total_label_slices -k-1))
#        logging.error('TEMP REM SHAPE : ' + str(temp_rem.shape))
        
        temp_label[:,:,int(total_label_slices -k-1)] = np.copy(temp_rem)
        
        rem = rem%(2**int(total_label_slices -k-1))
    return temp_label


def png_to_binary_with_ones(cropped_label):
    temp_label = np.ones((cropped_label.shape[0],cropped_label.shape[1],int(exp_config.num_classes+1)))
    
    rem = np.copy(cropped_label)
    for k in range(exp_config.num_classes):
        temp_label[:,:,int(exp_config.num_classes-k-1)] = rem//(2**int(exp_config.num_classes-k-1))
        
        rem = rem%(2**int(exp_config.num_classes-k-1))
    return temp_label

def inception_preprocess(image):
    image=np.float32(image)/255
    image = image - 0.5
    image = image*2
    
    return image


def inverse_inception_preprocess(image):
    image=np.float32(image)/2
    image = image + 0.5
    image = image*255
    
    return image

def write_to_txt_file(path, strings_list):
    file1 = open(path,"a") 
    for L in strings_list: 
        file1.write(L)
        file1.write("\n")
    file1.close()
   
    
def write_variables_to_txt_file(path, strings_list):
    file1 = open(path,"a") 
    for L in strings_list: 
        file1.write(str(L))
        file1.write("\n")
    file1.close()
    
def read_from_txt_file(path):
    with open(path) as t:
        txt = t.readlines()
    for k in range(len(txt)):
        if '\n' in txt[k]:
        
            txt[k] = txt[k][0:-1]
        
    return txt
def get_image_and_label(my_scene, ind):
    
    current_dir = os.path.join(target_dir,my_scene)
    

    
    all_images_list = sorted(glob.glob(os.path.join(current_dir,'img*.png')))
    all_labels_list = sorted(glob.glob(os.path.join(current_dir,'label*.png')))
    
    
    img = Image.open(all_images_list[ind])
    img.load()
    label = Image.open(all_labels_list[ind])
    label.load()
    
    image=np.array(img, dtype=np.uint8)

    label=np.array(label, dtype=np.uint8)
    
#    plt.imshow(image)
#    plt.imshow(label)
    
    
    return image,label
    
def get_cover_loss(gaussians,covers):
    labels=tf.to_float(tf.greater(covers,0.5))
    epsilon = 1e-05
    L=-(1-labels)*tf.log(1-gaussians + epsilon)
    return tf.reduce_mean(L)

def height_loss_func(cur_y, selected_heights):
    return tf.math.maximum(0.0,selected_heights - cur_y)
    
    
def run_training(continue_run):

#    train_file ='C:\\winpython\\WPy-3670\\codes\\davis2017\\DAVIS\\ImageSets\\2017\\train.txt'
#    data_images_path ='C:\\winpython\\WPy-3670\\codes\\davis2017\\DAVIS\\JPEGImages\\480p\\drone'
    
    
    logging.error('EXPERIMENT : ' + str(exp_config.experiment_name))
    logging.error('THIS IS : ' + str(log_dir))
    
    '''THIS IS DIFFERENT BETWEEN PRE TRAIN AND MAIN NETWORK. THIS ONLY USES STATIC IMAGES SO LIST OF ALL THE SAMPLES IS TRAIN SET
    WHILE MAIN NETWORK USES THE SCENES AS THE TRAIN SET
    '''
    
#    split = 0.9
#    
#    all_ids = np.arange(len(scenes))
#    val_scene_ids = all_ids[0::int(1/(1-split))]
#    train_scene_ids = np.setdiff1d(all_ids,val_scene_ids)
#    
#    
#    
#    N_train_videos = len(train_scene_ids)
#
#    logging.error('Total nmber of training samples : ' + str(N_train_videos))
#    batch_indices = np.copy(train_scene_ids)

    
#    val_tokens = read_from_txt_file('/home/cany/mapmaker/val_tokens.txt')
#    train_tokens = read_from_txt_file('/home/cany/mapmaker/train_tokens.txt')
    
    val_tokens = token_splits.VAL_SCENES
    train_tokens = token_splits.TRAIN_SCENES
    
    
    batch_indices = np.arange(len(train_tokens))
    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name)


    init_step = 0
    # Load data
    
   
    

    masks_list=[]
    occ_masks_list=[]
#        side_preds_list=[]

    # Tell TensorFlow that the model will be built into the default Graph.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
#        with tf.Graph().as_default():
    with tf.Session(config = config) as sess:
        # Generate placeholders for the images and labels.

        
        """         
        Note that the first frame mask is returned as float32 while the others as uint8. This is because the first frame mask
        is not used in loss calculations and only concatenated with the image in memory encoding. While the other masks
        are NOT used in memory encoding and only used in loss calculations. So for tf.one_hot to work they are uint8.
        """
        
#            memory_images_placeholder = tf.placeholder(tf.float32, shape=[None]+list(exp_config.image_size) + [3], name='mem_img_pl')
#            memory_labels_placeholder = tf.placeholder(tf.float32, shape=[None]+list(exp_config.image_size)+ [1], name='ref_label_pl')
#            
#            learning_rate_placeholder_transformer = tf.placeholder(tf.float32, shape=[])
#            learning_rate_placeholder_backbone = tf.placeholder(tf.float32, shape=[])
        learning_rate_placeholder = tf.placeholder(tf.float32, shape=[])
        training_time_placeholder = tf.placeholder(tf.bool, shape=[])
        
        my_training_placeholder = tf.placeholder(tf.bool, shape=[])
        

        # Build a Graph that computes predictions from the inference model.
        my_model_options = common.ModelOptions({common.OUTPUT_TYPE:10},crop_size=exp_config.camera_image_patch_size,atrous_rates=[6, 12, 18])
   
        image_tensor_shape = [n_frames_per_seq,exp_config.camera_image_patch_size[0],exp_config.camera_image_patch_size[1],3]
        image_mask_tensor_shape = [n_frames_per_seq,int(exp_config.camera_image_patch_size[0]/4),int(exp_config.camera_image_patch_size[1]/4),total_label_slices]
        mask_tensor_shape = [n_seqs,exp_config.patch_size[1],exp_config.patch_size[0],exp_config.num_bev_classes + 1]
        
        images_placeholder = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
        
        image_labels_placeholder = tf.placeholder(tf.float32, shape=image_mask_tensor_shape, name='image_labels')
        image_objects_label_placeholder= tf.placeholder(tf.float32, shape= [1,int(exp_config.camera_image_patch_size[0]/4),int(exp_config.camera_image_patch_size[1]/4),exp_config.num_object_classes+1], name='image_object_labels')
        covers_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,exp_config.patch_size[1],exp_config.patch_size[0],1], name='covers')
        
        separate_covers_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,exp_config.patch_size[1],exp_config.patch_size[0],1], name='separate_covers')
        
        bev_transforms_placeholder = tf.placeholder(tf.float32, shape=[np.max([1,n_seqs-1]),8], name='bev_transforms')
        
        ground_transforms_placeholder1 = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,8], name='ground_transforms1')
        
        ground_transforms_placeholder2 = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,8], name='ground_transforms2')
        
        ground_transforms_placeholder3 = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,8], name='ground_transforms3')
        
        
        coordinate_ground_transforms_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,3,3], name='coordinate_ground_transforms')
        
        channel_bev_images_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,exp_config.patch_size[1],exp_config.patch_size[0],3*num_frames], name='channel_images')
        
        
        ref_bev_labels_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,exp_config.label_patch_size[1],exp_config.label_patch_size[0],exp_config.num_bev_classes + 1], name='ref_bev_labels')          
        
        resized_covers = tf.image.resize(
                covers_placeholder, [int(exp_config.patch_size[1]/exp_config.bev_downsample_ratio),int(exp_config.patch_size[0]/exp_config.bev_downsample_ratio)] ,method='nearest',name='cover_resize'  )
        
    
        no_mask_tensor = tf.constant(-np.ones((1,int(exp_config.patch_size[1]/exp_config.feature_downsample),int(exp_config.patch_size[0]/exp_config.feature_downsample),int(exp_config.num_classes+1)),np.float32))
        
        '''
        Extract features from the CAMERA IMAGE
        '''
        
        image_total_backbone_out, image_total_relative_endpoints, image_total_end_points  =mem_net.image_encoder(images_placeholder,no_mask_tensor,my_model_options,downsample_stages=4,use_deeplab=use_deeplab,is_training=training_time_placeholder, reuse=False)
#            image_total_backbone_out = mem_net.my_image_decoder(image_total_relative_endpoints,image_total_backbone_out,reuse=False)
        
        
        total_input_image = image_total_backbone_out
        
        side_mask_logits,side_occ_est_logits, side_masks, side_occ_softmaxed = mem_net.my_side_decoder(image_total_relative_endpoints,total_input_image,reuse=False)
        
        reference_image_endpoints=[]
        for endi in range(len(image_total_relative_endpoints)):
            reference_image_endpoints.append(tf.slice(image_total_relative_endpoints[endi],[reference_frame_index,0,0,0],[1,-1,-1,-1]))
        
        side_obj_logits, side_obj_softmaxed = mem_net.my_object_side_decoder(reference_image_endpoints,tf.slice(total_input_image,[reference_frame_index,0,0,0],[1,-1,-1,-1]),exp_config,apply_softmax=True,reuse=False)
        # logging.error('SIDE OCC LOGITS ' + str(side_obj_))
        # logging.error('SIDE OCC LABELS ' + str(tf.squeeze(tf.slice(image_labels_placeholder,[0,0,0,exp_config.num_classes+1],[-1,-1,-1,-1]),axis=-1)))
        
        cur_covers = tf.slice(resized_covers,[0,0,0,0],[1,-1,-1,-1])
        
       
        projected_estimates = tf.contrib.image.transform(
            tf.concat([side_masks,side_occ_softmaxed],axis=-1),
            tf.squeeze(tf.slice(ground_transforms_placeholder1,[0,0,0],[1,-1,-1]),axis=0),
            interpolation='BILINEAR',
            output_shape=(exp_config.project_patch_size[1],exp_config.project_patch_size[0]),
            name='tensorflow_ground_transform'
        )
        
        cur_separate_covers = tf.squeeze(tf.slice(separate_covers_placeholder,[0,0,0,0,0],[1,-1,-1,-1,-1]),axis=0)
        combined_projected_estimates = tf.reduce_max(projected_estimates*cur_separate_covers,axis=0,keepdims=True)
        
        projected_obj_estimates = tf.contrib.image.transform(
            side_obj_softmaxed,
            tf.squeeze(tf.slice(ground_transforms_placeholder1,[0,reference_frame_index,0],[1,1,-1]),axis=0),
            interpolation='BILINEAR',
            output_shape=(exp_config.project_patch_size[1],exp_config.project_patch_size[0]),
            name='tensorflow_ground_transform'
        )
        
        combined_projected_estimates = tf.concat([combined_projected_estimates,projected_obj_estimates],axis=-1)
        resized_combined_projected_estimates = tf.image.resize(
                combined_projected_estimates, [int(exp_config.patch_size[1]/8),int(exp_config.patch_size[0]/8)] ,method='bilinear',name='projected_estimates_resize'  )
        
        
        '''
        Scale the coordinates to the original image so that the transformation is compatible
        '''
     
        
        all_bev_total_backbone_out = tf.contrib.image.transform(
            image_total_relative_endpoints[0],
            tf.squeeze(tf.slice(ground_transforms_placeholder2,[0,0,0],[1,-1,-1]),axis=0),
            interpolation='BILINEAR',
            output_shape=(exp_config.project_patch_size[1],exp_config.project_patch_size[0]),
            name='tensorflow_ground_transform_end1'
        )
        cur_separate_covers = tf.squeeze(tf.slice(separate_covers_placeholder,[0,0,0,0,0],[1,-1,-1,-1,-1]),axis=0)
        # combined_back_out = tf.reduce_max(tf.slice(all_bev_total_backbone_out,[0,0,0,0],[-1,-1,-1,128])*cur_separate_covers,axis=0,keepdims=True)
        
        # combined_back_out = tf.concat([tf.tile(combined_back_out,[num_frames,1,1,1]),tf.slice(all_bev_total_backbone_out,[0,0,0,128],[-1,-1,-1,-1])],axis=-1)
        
        combined_back_out = tf.reduce_max(all_bev_total_backbone_out*cur_separate_covers,axis=0,keepdims=True)
        
        combined_back_out = tf.concat([combined_back_out,tf.slice(all_bev_total_backbone_out,[reference_frame_index,0,0,0],[1,-1,-1,-1])],axis=-1)
        
        
        bev_total_backbone_out = tf.image.resize(
                combined_back_out, [int(exp_config.patch_size[1]/8),int(exp_config.patch_size[0]/8)] ,method='bilinear',name='projected_estimates_resize'  )
        
        all_bev_end2 = tf.contrib.image.transform(
            image_total_relative_endpoints[1],
            tf.squeeze(tf.slice(ground_transforms_placeholder1,[0,0,0],[1,-1,-1]),axis=0),
            interpolation='BILINEAR',
            output_shape=(exp_config.project_patch_size[1],exp_config.project_patch_size[0]),
            name='tensorflow_ground_transform_end2'
        )
        
        logging.error('ENDPOINT WARPED ' + str(all_bev_end2))
        
        
        cur_separate_covers = tf.squeeze(tf.slice(separate_covers_placeholder,[0,0,0,0,0],[1,-1,-1,-1,-1]),axis=0)
        
        # combined_end = tf.reduce_max(tf.slice(all_bev_end2,[0,0,0,0],[-1,-1,-1,128])*cur_separate_covers,axis=0,keepdims=True)
        
        # combined_end = tf.concat([tf.tile(combined_end,[num_frames,1,1,1]),tf.slice(all_bev_end2,[0,0,0,128],[-1,-1,-1,-1])],axis=-1)
        
        combined_end = tf.reduce_max(all_bev_end2*cur_separate_covers,axis=0,keepdims=True)
        
        combined_end = tf.concat([combined_end,tf.slice(all_bev_end2,[reference_frame_index,0,0,0],[1,-1,-1,-1])],axis=-1)
        
        # combined_end = tf.reduce_max( all_bev_end2*cur_separate_covers,axis=0,keepdims=True)
        
        combined_end = tf.image.resize(
                combined_end, [int(exp_config.patch_size[1]/4),int(exp_config.patch_size[0]/4)] ,method='bilinear',name='projected_estimates_resize'  )
        
        bev_total_relative_endpoints = [combined_end]
        
       
        
        total_input = tf.concat([ resized_combined_projected_estimates,bev_total_backbone_out],axis=-1)
        
        
        
        
        static_logits, static_masks,object_logits, object_masks = mem_net.my_bev_object_decoder(bev_total_relative_endpoints,total_input,exp_config,apply_softmax=True,reuse=False)
        masks = tf.concat([static_masks,object_masks],axis=-1)
        
        cur_covers = tf.slice(resized_covers,[0,0,0,0],[1,-1,-1,-1])
        
      
        
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        trainable_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        optimizer_variables=[]

        backbone_optimizer_variables=[]
        
        load_new_vars = []
        to_load_vars = []
        
#        for var in all_vars:
#                    
#          if 'side' in var.op.name:
#              load_new_vars.append(var)
#              
#              
#          else:
#              to_load_vars.append(var)
              
        if use_deeplab:
            if starting_from_cityscapes:
                query_variables_to_restore = dict()

                restored_vars = []
                new_vars=[]

                for var in all_vars:
                    
                  if 'image_encoder' in var.op.name:
                      restored_vars.append(var.op.name)
                      
                      if 'mem_net_backbone' in var.op.name:
                      
                          query_variables_to_restore[var.op.name.replace('image_encoder/mem_net_backbone/', '')]=var
                      elif 'pretrained_decoder' in var.op.name:
                          query_variables_to_restore[var.op.name.replace('memory_encoder/pretrained_decoder/', '')]=var
                      
                      
                  else:
                      new_vars.append(var)
#                      trainable_new_vars.append(var)
                imagenet_query_saver = tf.train.Saver(query_variables_to_restore)
            
        else:
            image_variables_to_restore = dict()
            bev_variables_to_restore = dict()
            restored_vars = []
            new_vars=[]
            for var in all_vars:
                    
              if 'resnet_backbone' in var.op.name:
                  restored_vars.append(var.op.name)
                  
                  if 'image_encoder' in var.op.name:
                      image_variables_to_restore[var.op.name.replace('image_encoder/resnet_backbone/', '')]=var
#                      else:
#                          if not ('backbone_combine_conv' in var.op.name):
#                              bev_variables_to_restore[var.op.name.replace('bev_encoder/resnet_backbone/', '')]=var
              else:
                  new_vars.append(var)
#                      trainable_new_vars.append(var)
            imagenet_image_saver = tf.train.Saver(image_variables_to_restore)
#                imagenet_bev_saver = tf.train.Saver(bev_variables_to_restore)
            
            
        non_decoder_vars = []
        for var in trainable_vars:
            if not ('decoder' in var.op.name):
                non_decoder_vars.append(var)
        
        for var in trainable_vars:
            
            if 'BatchNorm' in var.op.name:
#                        print('Batch norm variable : '  + str(var))
                continue
            
            
            
            elif 'upscale' in var.op.name:
                print('Upscale variable : ' + str(var))
                
            elif 'bev_encoder' in var.op.name:
                backbone_optimizer_variables.append(var)
                optimizer_variables.append(var)
            elif 'image_encoder' in var.op.name:
                
                if use_deeplab:
                    if 'exit' in var.op.name:
                        backbone_optimizer_variables.append(var)
                        optimizer_variables.append(var)
                else:
                    backbone_optimizer_variables.append(var)
                    optimizer_variables.append(var)
                    logging.error('BACKBONE VAR' + str(var))
#                    elif 'conv1/biases' in var.op.name:
#                        print('Conv1 biases are initialized as zero and we use batch norm on the sum of conv1 and mask conv')
#                    
#                    elif 'conv1/biases' in var.op.name:
#                        print('Conv1 biases are initialized as zero and we use batch norm on the sum of conv1 and mask conv')
#                    
                    
            else:
                logging.error('NON BACKBONE VAR ' + str(var))
                optimizer_variables.append(var)
        
        
        logging.error('NUMBER OF ALL PARAMETERS: ' + str(np.sum([np.prod(v.get_shape().as_list()) for v in optimizer_variables])))
        logging.error('NUMBER OF BACKBONE PARAMETERS: ' + str(np.sum([np.prod(v.get_shape().as_list()) for v in backbone_optimizer_variables])))
        
        
#        to_load_saver = tf.train.Saver(var_list=to_load_vars,max_to_keep=2)
        saver = tf.train.Saver(max_to_keep=2)
        
        # saver_best_loss = tf.train.Saver(max_to_keep=2)
        init = tf.global_variables_initializer()
        sess.run(init)
        
#        load_path = os.path.join('/scratch_net/catweazle/cany/mapmaker_bev_object/logdir/deeplab'+str(use_deeplab),'checkpoints/routine-59999')
        load_path = os.path.join('/scratch_net/catweazle/cany/mapmaker_bev_object/logdir/deeplab'+str(use_deeplab),'checkpoints/save','routine-39999')
        # load_path = os.path.join('/scratch_net/catweazle/cany/mapmaker_bev_object/logdir/deeplab'+str(use_deeplab),'checkpoints','routine-84999')

##            load_path = os.path.join('/scratch_net/catweazle/cany/mapmaker_cond/logdir/use_occlusion'+str(use_occlusion)+'deeplab'+str(use_deeplab), exp_config.experiment_name,'loop'+str(loop),'Recs'+str(rec_count)+'intermediateLoss'+str(apply_intermediate_loss),'checkpoints','routine-39999')
##            load_path = os.path.join('/scratch_net/catweazle/cany/mapmaker_cond/logdir/deeplabFalse/nuscenes_exp/loopTrue/Recs2intermediateLossFalse','checkpoints','routine-39999')
###                load_path = os.path.join('/scratch_net/catweazle/cany/mapmaker_cond/logdir/', exp_config.experiment_name,'loop'+str(loop),'Recs'+str(rec_count)+'intermediateLoss'+str(apply_intermediate_loss),'checkpoints','routine-19999')
##
        saver.restore(sess,load_path)
#            to_load_saver.restore(sess,load_path)
        sess.run(mem_net.interp_surgery(tf.global_variables()))
        
    
        val_res=do_eval(sess,val_tokens,
                             my_training_placeholder,
        
         images_placeholder,
      
         image_labels_placeholder,
         covers_placeholder,
         bev_transforms_placeholder,
         separate_covers_placeholder,
         ground_transforms_placeholder1,
         ground_transforms_placeholder2,
         ground_transforms_placeholder3,
         coordinate_ground_transforms_placeholder,
        
         channel_bev_images_placeholder,
        
         masks,
         side_masks,side_occ_softmaxed,side_obj_softmaxed,
         projected_estimates,
         projected_obj_estimates,
         combined_projected_estimates,
        0,training_time_placeholder,val_folder_path=validation_res_path)
        overall_mean = np.mean(np.array(val_res))
        logging.error('Overall mean : ' + str(overall_mean))

def save_keypoints_array(array,name,to_size=None,val=True):
    
    res_ar = np.max(array,axis=-1)
#    logging.error('TO SAVE '+ name + '  , '+str(res_ar.shape))
    
#        
#        temp_list = []
#        for k in range(res_ar.shape[0]):
#    
#            temp_ar = res_ar[k,...]
#            temp_ar = cv2.resize(temp_ar,to_size, interpolation = cv2.INTER_LINEAR)
#            temp_list.append(temp_ar)
#        res_ar = np.stack(temp_list,axis=0)
#    logging.error('TO SAVE RESZIED '+ name + '  , '+str(res_ar.shape))    
#        
    for k in range(res_ar.shape[0]):
        
         cur_slice = np.squeeze(res_ar[k,...])
         
         if to_size is not None:
             cur_slice = cv2.resize(cur_slice,to_size, interpolation = cv2.INTER_LINEAR)
                     
         img_png=Image.fromarray(np.uint8(255*cur_slice))
         
         img_png.save(os.path.join(train_results_path,name+'_batch_'+str(k)+'.jpg'))
         
    
    
    
def save_array(array,name,slice_last_dim=True,is_rgb=False,to_size=None,correct=True,val=False):
     
    
        
     if is_rgb:
         if correct:
             if not use_deeplab:
                 array = array + means_image
     for k in range(array.shape[0]):
        
         cur_slice = np.squeeze(array[k,...])
         if to_size is not None:
             if is_rgb:
                 cur_slice = cv2.resize(cur_slice,to_size, interpolation = cv2.INTER_LINEAR)
             else:
                 cur_slice = cv2.resize(cur_slice,to_size, interpolation = cv2.INTER_NEAREST)
         
         if is_rgb:
             img_png=Image.fromarray(np.uint8(cur_slice))
             
             if val:
                 img_png.save(os.path.join(validation_res_path,name+'_'+str(k)+'.jpg'))
             else:
                img_png.save(os.path.join(train_results_path,name+'_'+str(k)+'.jpg'))
         else:
             
             if slice_last_dim:
                 for m in range(cur_slice.shape[-1]):
                     
                     img_png=Image.fromarray(np.uint8(255*cur_slice[...,m]))
                     if val:
                        img_png.save(os.path.join(validation_res_path,name+'_batch_'+str(k)+'_class_'+str(m)+'.jpg'))
                     else:
                        img_png.save(os.path.join(train_results_path,name+'_batch_'+str(k)+'_class_'+str(m)+'.jpg'))
                     
             else:
                 
                 
                img_png=Image.fromarray(np.uint8(cur_slice*255))
                if val:
                    img_png.save(os.path.join(validation_res_path,name+'_'+str(k)+'.jpg'))
                else:
                    img_png.save(os.path.join(train_results_path,name+'_'+str(k)+'.jpg'))
         
def single_image_vgg_preprocess(orig_img, orig_label):
    means_image = np.array([123.68, 116.779, 103.939], dtype=np.single)
    original_size = orig_label.size
    
   
    orig_img=np.array(orig_img, dtype=np.float32)- means_image
    
    
    orig_label = np.array(orig_label, dtype=np.uint8)
    orig_label = png_to_binary_with_ones(orig_label)
    
    to_feed_image = np.zeros((912,1600,3))
    to_feed_label = np.zeros((912,1600,orig_label.shape[-1]))
    
    to_feed_image[6:906,:,:] = orig_img
    to_feed_label[6:906,:,:] = orig_label
    to_feed_label[...,-1] = 1
    
    return original_size,orig_img, orig_label, to_feed_image, to_feed_label

def write_results_to_folder(name_of_seq,frame_number,to_eval_estimates):
    
    root_folder = os.path.join(validation_res_path,name_of_seq)
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    len_of_number = len(str(frame_number))
    init_str = str(frame_number)
    for k in range(5-len_of_number):
        init_str = '0'+init_str
    img_png=Image.fromarray(to_eval_estimates.astype(np.uint8))
    img_png.save(os.path.join(root_folder,init_str+'.png'))    
    
def  single_image_inception_preprocess(orig_img, orig_label):
    
    orig_img = np.array(orig_img, dtype=np.uint8)
    
    orig_label = np.array(orig_label, dtype=np.uint8)

    new_sizes = (exp_config.patch_size[1],exp_config.patch_size[0])
#    cropped_label = cv2.resize(orig_label,new_sizes, interpolation = cv2.INTER_NEAREST)
    cropped_img = cv2.resize(orig_img,new_sizes, interpolation = cv2.INTER_LINEAR)
    
    temp_label = np.zeros((orig_label.shape[0],orig_label.shape[1],total_label_slices))
    
    rem = np.copy(orig_label)
#    logging.error('Rem shape ' + str(rem.shape))
    for k in range(total_label_slices):
        
        temp_rem = rem//(2**int(total_label_slices-k-1))
#        logging.error('TEMP REM SHAPE : ' + str(temp_rem.shape))
        
        temp_label[:,:,int(total_label_slices-k-1)] = np.copy(temp_rem)
        
        rem = rem%(2**int(total_label_slices-k-1))
        
#    pre_img=cropped_img- means_image
#    pre_img = inception_preprocess(cropped_img)

    if not use_deeplab:
        pre_img = cropped_img - means_image
    else:
        pre_img=cropped_img
#    pre_img = cropped_img
    
    return orig_img, orig_label   ,pre_img  ,temp_label


def get_label_mask(ar):
    
    vis_mask = ar[...,exp_config.num_classes]
    occ_mask = ar[...,exp_config.num_classes+1]
    
    tot_mask = occ_mask*vis_mask
    
    return tot_mask
    


def eval_iterator(my_scene,cur_index, reference_frame_index, single_frame=False,apply_interval=False): 
    n_seqs = 1
    current_dir = os.path.join(target_dir,'scene'+my_scene)
    
#        logging.error('Cur directory : ' + str(current_dir))
    pool = ThreadPool(n_seqs*num_frames) 

    
    all_images_list = sorted(glob.glob(os.path.join(current_dir,'img*.png')))
    all_labels_list = sorted(glob.glob(os.path.join(current_dir,'label*.png')))
    
#        logging.error(str(all_images_list))
#        logging.error(str(os.listdir(current_dir)))
    
    first_frame = cur_index
    
#        for frame_number in range(1,n_frames_per_seq):
#            frame_ids.append(random.randint(frame_ids[-1]+1, np.min([n_frames_in_scene-(n_frames_per_seq-frame_number),np.max([frame_ids[-1]+1,n_frames_in_scene-(n_frames_per_seq-frame_number-1)*2-1]),frame_ids[-1]+max_interval_between_frames])))
#            

    frame_ids=[]
    frame_ids.append(first_frame)
    
    # logging.error('LEN IMAGES ' + str(len(all_images_list)))
    
    if single_frame:
    
        for frame_number in range(1,n_frames_per_seq):
            frame_ids.append(first_frame )
    else:
        if apply_interval:
            for frame_number in range(1,n_frames_per_seq):
                frame_ids.append(first_frame + frame_interval*frame_number)
            
        else:    
            for frame_number in range(1,n_frames_per_seq):
                frame_ids.append(first_frame + frame_number)
        
    pairs = []
    
    
    scene_token = current_dir.split('/')[-1][5:]
    
    my_scene = nusc.get('scene', scene_token)
    
    first_sample_token = my_scene['first_sample_token']
    last_sample_token = my_scene['last_sample_token']
    first_sample_ind = nusc.getind('sample',first_sample_token)
    last_sample_ind = nusc.getind('sample',last_sample_token)
    
    all_sample_inds = np.arange(first_sample_ind,last_sample_ind+1)
    
    transforms_list=[]        
    reference_samples = []
    
#        logging.error('STARTING BEV TO BEV')
    
    bev_labels_list=[]
    
    for k in range(n_seqs):
        cur_ref_sample = nusc.sample[all_sample_inds[frame_ids[k+reference_frame_index]]]
        reference_samples.append(cur_ref_sample)
        
        cam_token_cur = cur_ref_sample['data']['CAM_FRONT']
        cam_record_cur = nusc.get('sample_data', cam_token_cur)
        
        bev_label = np.array(Image.open( os.path.join('/srv/beegfs02/scratch/tracezuerich/data/cany/monomaps_labels_vanilla',  
                                   cam_record_cur['token'] + '.png')),np.int32)
        
        bev_label = np.flipud(bev_label)
        
        bev_label = decode_binary_labels(bev_label,exp_config.num_bev_classes+1)

        bev_labels_list.append(bev_label)
        cs_record_cur = nusc.get('calibrated_sensor', cam_record_cur['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_record_cur['camera_intrinsic'])
        
        # np.savez('/home/cany/image_trans_stuff.npy',cs_record_cur,cam_intrinsic)
        
        to_image_transform = project_to_image(np.zeros((exp_config.project_base_patch_size[1],exp_config.project_base_patch_size[0])),cs_record_cur,cam_intrinsic)

        
#        logging.error('BEV TO BEV ENDED')
    

    
#        my_sample = nusc.sample[scene_samples[m]]
    
    
    if single_frame:
    
        for k in range(n_seqs):
            for m in range(num_frames):
                pairs.append((all_images_list[frame_ids[k + m]],all_labels_list[frame_ids[k+m]],reference_samples[k],nusc.sample[all_sample_inds[frame_ids[k+m]]],True))
            
    else:    
        for k in range(n_seqs):
            for m in range(num_frames):
                pairs.append((all_images_list[frame_ids[k + m]],all_labels_list[frame_ids[k+m]],reference_samples[k],nusc.sample[all_sample_inds[frame_ids[k+m]]],m==reference_frame_index))
        

    results = pool.map(single_process,pairs)
    
    
    pool.close() 
    pool.join() 
#        logging.error('Results shape : ' + str(len(results)))

    seq_images_ar=np.zeros((n_frames_per_seq,exp_config.camera_image_patch_size[0],exp_config.camera_image_patch_size[1],3),np.float32)
    seq_labels_ar=np.ones((n_frames_per_seq,int(exp_config.camera_image_patch_size[0]/4),int(exp_config.camera_image_patch_size[1]/4),int(total_label_slices  )),np.float32)

    bev_transforms_ar1=np.ones((n_seqs,num_frames,8),np.float32)
    bev_transforms_ar2=np.ones((n_seqs,num_frames,8),np.float32)
    bev_transforms_ar3=np.ones((n_seqs,num_frames,8),np.float32)
    
    coordinate_transforms_ar=np.ones((n_seqs,num_frames,3,3),np.float32)
    bev_images_ar=np.zeros((n_seqs,num_frames,exp_config.patch_size[1],exp_config.patch_size[0],3),np.float32)
    bev_labels_ar=np.ones((n_seqs,num_frames,exp_config.patch_size[1],exp_config.patch_size[0],int(total_label_slices  )),np.float32)
    bev_covers_ar=np.ones((n_seqs,num_frames,exp_config.patch_size[1],exp_config.patch_size[0],1),np.float32)

#        logging.error('PROJECT TO GROUND ENDED')
    for k in range(len(results)):
        temp_res = results[k]
        
        if k < num_frames:
        
            seq_images_ar[k,...] = np.copy(temp_res[0])
            seq_labels_ar[k,...] = np.copy(temp_res[1])
       
        elif k >= (n_seqs*num_frames - (n_seqs - 1)):
            seq_images_ar[k - (num_frames-1),...] = np.copy(temp_res[0])
            seq_labels_ar[k - (num_frames-1),...] = np.copy(temp_res[1])
        
#                logging.error('RETURNED GRID SHAPE ' + str(temp_res[3].shape))
        bev_images_ar[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[2])
        bev_labels_ar[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[3])
        bev_covers_ar[int(k//num_frames),k%num_frames,...] = np.expand_dims(np.copy(temp_res[4][...,0]),axis=-1)
        
        bev_transforms_ar1[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[6])
        bev_transforms_ar2[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[7])
        bev_transforms_ar3[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[8])
        coordinate_transforms_ar[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[5])
   
    

    return seq_images_ar, seq_labels_ar,bev_images_ar,bev_labels_ar,bev_covers_ar,np.zeros((1,8)), bev_transforms_ar1,bev_transforms_ar2,bev_transforms_ar3,coordinate_transforms_ar,np.stack(bev_labels_list,axis=0),to_image_transform,True


def overall_eval_iterator(my_scene,cur_index, reference_frame_index, single_frame=False,apply_interval=False):
    
    # logging.error('SINGLE FRAME ' + str(single_frame))
    seq_images_ar, seq_labels_ar, bev_images_ar,bev_labels_ar,bev_covers_ar, transforms_ar,tf_transforms1,tf_transforms2,tf_transforms3, coordinate_transforms_ar,real_ref_bev_labels,to_image_transform,went_well =  eval_iterator(my_scene,cur_index,reference_frame_index,single_frame=single_frame,apply_interval=apply_interval)
        

    squeezed_bev_covers_ar = np.squeeze(bev_covers_ar,axis=-1)
    
    total_img_list=[]
    total_labels_list=[]

    for k in range(n_seqs):
        total_img = np.zeros_like(bev_images_ar[0,0,...])
        total_labels = np.zeros_like(bev_labels_ar[0,0,...])
        for m in range(num_frames):
            total_img[squeezed_bev_covers_ar[k,m,...]>0.5,:] = bev_images_ar[k,m,...][squeezed_bev_covers_ar[k,m,...]>0.5,:]
            total_labels[squeezed_bev_covers_ar[k,m,...]>0.5,:] = bev_labels_ar[k,m,...][squeezed_bev_covers_ar[k,m,...]>0.5,:]
            
        total_img_list.append(total_img)
        total_labels_list.append(total_labels)
    
    fin_bev_images = np.stack(total_img_list,axis=0)
    fin_bev_labels = np.stack(total_labels_list,axis=0)
    fin_covers = np.clip(np.sum(bev_covers_ar,axis=1),0,1)
    
    my_area = np.float32(bev_covers_ar > 0.5)
    
#        logging.error('BEV IMAGES MAX ' + str(np.max(bev_images_ar))+ ' MIN ' + str(np.min(bev_images_ar)))
    
    separate_bev_images = my_area*bev_images_ar
    
    to_return_bev_images_list = []
    
    for k in range(num_frames):
        to_return_bev_images_list.append(separate_bev_images[:,k,...])
        
    to_return_bev_images = np.concatenate(to_return_bev_images_list,axis=-1)
   
    image_objects = cv2.warpPerspective(np.squeeze(real_ref_bev_labels[...,4:-1]),to_image_transform,exp_config.original_image_size,flags=cv2.INTER_LINEAR)
    image_objects= cv2.resize(image_objects,(int(exp_config.camera_image_patch_size[1]/4),int(exp_config.camera_image_patch_size[0]/4)), interpolation = cv2.INTER_LINEAR)
    image_objects = np.expand_dims(np.float32(image_objects > 0.5),axis=0)
    
    image_objects = np.concatenate([image_objects,np.clip(1-np.sum(image_objects,axis=-1,keepdims=True),0,1)],axis=-1)
    
        
    return seq_images_ar, seq_labels_ar, fin_bev_images, fin_bev_labels,fin_covers , transforms_ar,tf_transforms1,tf_transforms2,tf_transforms3, bev_covers_ar, coordinate_transforms_ar,to_return_bev_images,real_ref_bev_labels,image_objects




def do_eval(sess,val_tokens,
                                            my_training_placeholder,
                        
                        images_placeholder,
                        
                        image_labels_placeholder,
                        covers_placeholder,
                        bev_transforms_placeholder,
                        separate_covers_placeholder,
                        ground_transforms_placeholder1,
                        ground_transforms_placeholder2,
                        ground_transforms_placeholder3,
                        coordinate_ground_transforms_placeholder,
                        
                        channel_bev_images_placeholder,
                        
                        masks,
                        side_masks,side_occ_masks,side_obj_softmaxed,
                        projected_estimates,
                       projected_obj_estimates,
                        combined_projected_estimates,
                        iteration,training_time_placeholder,val_folder_path=validation_res_path):

    '''
    Function for running the evaluations every X iterations on the training and validation sets. 
    :param sess: The current tf session 
    :param eval_loss: The placeholder containing the eval loss
    :param images_placeholder: Placeholder for the images
    :param labels_placeholder: Placeholder for the masks
    :param training_time_placeholder: Placeholder toggling the training/testing mode. 
    :param images: A numpy array or h5py dataset containing the images
    :param labels: A numpy array or h45py dataset containing the corresponding labels 
    :param batch_size: The batch_size to use. 
    :return: The average loss (as defined in the experiment), and the average dice over all `images`. 
    '''
    logging.error('Started evaluation')
    
    
    res_strings=[]
    all_j1s=[]
    occ_all_j1s=[]
    framewise_best=[]
    # figure_tokens=['ed97ee6a3b444ba7800641baab057556',
    #                '373bf99c103d4464a7b963a83523fbcb','6d4b2bd795ae4c66900ad98ccd2371a6',
    #                '4f18d9a7ed374a0fb93c026589dcf9a0']
    # ed97ee6a3b444ba7800641baab057556_img00005
    # 373bf99c103d4464a7b963a83523fbcb_img00005
    # 6d4b2bd795ae4c66900ad98ccd2371a6_img00003
    # 4f18d9a7ed374a0fb93c026589dcf9a0_img00029
    counter = 0
    increments = [-0.2,-0.15,-0.1,-0.05,0,0.05,0.1,0.15,0.2]
    for my_scene_token in val_tokens:
        counter = counter + 1
        logging.error('SCENE ' + str(counter) + ' of ' + str(len(val_tokens)) + ' scenes')
        
        scene_results=[]
        occ_scene_results=[]
        current_dir = os.path.join(target_dir,'scene'+my_scene_token)
        frame_numbers_list=[]
        images = sorted(glob.glob(os.path.join(current_dir,'img*.png')))
        labels = sorted(glob.glob(os.path.join(current_dir,'label*.png')))
        name_of_seq = my_scene_token
        
        scene_token = current_dir.split('/')[-1][5:]
        
        my_scene = nusc.get('scene', scene_token)
        
        if not os.path.exists(os.path.join(validation_res_path,name_of_seq)):
            os.makedirs(os.path.join(validation_res_path,name_of_seq))
        
        
        
        for frame_number in range(len(images)):
            logging.error('FRAME NUMBER ' + str(frame_number))
            

#            logging.error('FRAME NUMBER ' + str(frame_number))
            if single_frame_test:
            
                batch_image, batch_label, batch_bev_images, batch_bev_labels,batch_bev_covers ,batch_transforms,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels,image_objects =\
                overall_eval_iterator(my_scene_token,frame_number,reference_frame_index,single_frame=True,apply_interval=False)
                    
            else:
                if frame_number < reference_frame_index:
                    
                
                    batch_image, batch_label, batch_bev_images, batch_bev_labels,batch_bev_covers ,batch_transforms,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels,image_objects =\
                    overall_eval_iterator(my_scene_token,frame_number,reference_frame_index,single_frame=True,apply_interval=False)
                    
                elif frame_number < frame_interval*reference_frame_index:
                    
                    batch_image, batch_label, batch_bev_images, batch_bev_labels,batch_bev_covers ,batch_transforms,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels ,image_objects=\
                    overall_eval_iterator(my_scene_token,int(frame_number - reference_frame_index),reference_frame_index,single_frame=False,apply_interval=False)
                    
                elif (frame_number >= (len(images) - (num_frames - reference_frame_index - 1))):
                
                    batch_image, batch_label, batch_bev_images, batch_bev_labels,batch_bev_covers ,batch_transforms,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels,image_objects =\
                    overall_eval_iterator(my_scene_token,frame_number,reference_frame_index,single_frame=True,apply_interval=False)
                    
                elif (frame_number >= (len(images) - frame_interval*(num_frames - reference_frame_index - 1))):
                
                    batch_image, batch_label, batch_bev_images, batch_bev_labels,batch_bev_covers ,batch_transforms,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels,image_objects =\
                    overall_eval_iterator(my_scene_token,int(frame_number - reference_frame_index),reference_frame_index,single_frame=False,apply_interval=False)
                        
                else:
                    
                    batch_image, batch_label, batch_bev_images, batch_bev_labels,batch_bev_covers ,batch_transforms,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels,image_objects =\
                    overall_eval_iterator(my_scene_token,int(frame_number - frame_interval*reference_frame_index),reference_frame_index,single_frame=False,apply_interval=True)
                    
            # class_label = np.sum(np.squeeze(batch_ref_bev_labels),axis=(0,1))
            # logging.error('CLASS LABELS ' + str(class_label))
            # if not (np.sum(class_label > 10) > 5):
            #     # logging.error('PASSED FRAME ' + str(frame_number))
            #     continue
            # frame_numbers_list.append(frame_number)
        
            feed_dict = {
                                
           
            training_time_placeholder: False,
            my_training_placeholder:False,
            
            images_placeholder:batch_image,
              
            covers_placeholder:batch_bev_covers,
            bev_transforms_placeholder:batch_transforms,
            separate_covers_placeholder : batch_separate_covers,
            ground_transforms_placeholder1:batch_tf_transforms1,
            ground_transforms_placeholder2:batch_tf_transforms2,
            ground_transforms_placeholder3:batch_tf_transforms3,
            coordinate_ground_transforms_placeholder:batch_coordinate_transforms,
            
            channel_bev_images_placeholder:batch_channel_bev_images,
            
            
            }

            masks_v,side_masks_v,side_occ_masks_v, projected_estimates_v,combined_projected_estimates_v,projected_obj_v,side_obj_softmaxed_v = sess.run([masks,
                        side_masks,side_occ_masks,projected_estimates,projected_obj_estimates,
                        combined_projected_estimates,side_obj_softmaxed], feed_dict=feed_dict)
#            logging.error('DEC OUTPUT ' + str(temp_dec_output.shape))
            
            
        
            grid_j1s=[]
        
       
    #]
#            my_thresh = np.expand_dims(np.expand_dims(np.array([0.5,0.5,0.45,0.3,
#                                                                0.5, 0.5, 0.5, 0.3,
#                                                                0.3, 0.6, 0.45, 0.4,
#                                                                0.45,0.5]),axis=0),axis=0)
#            thresh_list=[0.5,0.5,0.45,0.35,
#                         0.5,0.5,0.5,0.3,
#                         0.3,0.6,0.45,0.45,
#                         0.45,0.5]
            
            thresh_list=[0.5,0.5,0.5,0.5,
                         0.5,0.5,0.5,0.5,
                         0.5,0.5,0.5,0.5,
                         0.5,0.5]
            
            thresh_list=[0.5,0.5,0.45,0.3,
                                            0.5, 0.5, 0.5, 0.3,
                                            0.3, 0.6, 0.45, 0.4,
                                            0.45,0.5]
           
            # for inc in increments:
                
                
                
                
            squeezed_masks = np.squeeze(masks_v)
            
            
            hard_estimates = np.zeros((squeezed_masks.shape[0],squeezed_masks.shape[1],14))
            
            for k in range(exp_config.num_bev_classes):
                hard_estimates[...,k] = np.uint8(squeezed_masks[...,k] > (thresh_list[k]))
        
            
#            hard_estimate_list = []
            sample_results=[]
#            
            # thresh_grid = np.tile(np.expand_dims(np.linspace(0.6,0.4,196),axis=-1),[1,200])
            
            
#            thresh_list = 0.4*np.ones((8,1))
            for k in range(exp_config.num_bev_classes):
                # if k==1:
#                bev_estimate = np.squeeze(masks_v[...,k])
#                hard_estimate_list.append(np.uint8(bev_estimate > thresh_list[k]))
                # else:
                #     bev_estimate = np.squeeze(masks_v[...,k])
                #     hard_estimate_list.append(np.uint8(bev_estimate > thresh_grid))
                all_stats , void_pixels= get_all_stats(np.squeeze(batch_ref_bev_labels[...,k]), hard_estimates[...,k],np.squeeze(batch_ref_bev_labels[...,exp_config.num_bev_classes]),mask_iou=exp_config.use_occlusion)
                sample_results.append(all_stats)
            
            grid_j1s.append(np.array(sample_results))
       
            # logging.error('CALCS MADE')
            # save_array(np.expand_dims(np.uint8(void_pixels),axis=0),name_of_seq+'/void'+str(frame_number),slice_last_dim=False,is_rgb=False,val=True)
                 
            # save_keypoints_array(my_gaussians_v,name_of_seq+'/bev_keypoints'+str(frame_number),to_size=exp_config.patch_size,val=True)
            
#            for k in range(num_frames):
            # save_array(inverse_inception_preprocess(batch_channel_bev_images[...,2*3:3*(2+1)]) ,name_of_seq+'/channel_images_'+str(frame_number),slice_last_dim=False,is_rgb=True,correct=False,val=True)
            
#            save_array(batch_image  ,name_of_seq+'/orig_images'+str(frame_number),slice_last_dim=False,is_rgb=True,val=True)
#            save_array(batch_image[ref_ind:ref_ind+1]  ,name_of_seq+'/orig_images'+str(frame_number),slice_last_dim=False,is_rgb=True,val=True)
            
            # save_array(batch_label,name_of_seq+'/orig_labels'+str(frame_number),slice_last_dim=True,is_rgb=False,val=True)
            
            # save_keypoints_array(keypoint_image_coords_converted_v,name_of_seq+'/image_keypoints'+str(frame_number),to_size=(exp_config.original_image_size[0],exp_config.original_image_size[1]),val=True)
            
#            save_array(inverse_inception_preprocess(batch_bev_images)  ,'bev_images',slice_last_dim=False,is_rgb=True,correct=False)
            
#            save_array(batch_bev_labels,'bev_labels',slice_last_dim=True,is_rgb=False)
            
            # save_array(batch_ref_bev_labels,name_of_seq+'/real_labels_'+str(frame_number),slice_last_dim=True,is_rgb=False,val=True)
            
#            save_array(batch_bev_covers,'bev_covers',slice_last_dim=False,is_rgb=False)
            
#                            transformed_estimate_v[transformed_estimate_v < 0] = 0
#                            logging.error('TRANSFORMED ESTIMATE ' + str(transformed_estimate_v.shape))
#                            save_array(transformed_estimate_v,'transformed_estimate',slice_last_dim=True,is_rgb=False)
            
            # save_array(masks_v,name_of_seq+'/masks_'+str(frame_number),slice_last_dim=True,is_rgb=False,val=True)
            
#            save_array(np.float32(np.expand_dims(np.stack(hard_estimate_list,axis=-1),axis=0)),name_of_seq+'/hard_masks_'+str(frame_number),slice_last_dim=True,is_rgb=False,val=True)
#            
            # save_array(np.float32(np.expand_dims(hard_estimates,axis=0)),name_of_seq+'/hard_masks_'+str(frame_number),slice_last_dim=True,is_rgb=False,val=True)
            
            # logging.error('SIDE MASKS ' + str(side_masks_v.shape))
            
            # save_array(side_masks_v,name_of_seq+'/side_masks_'+str(frame_number),slice_last_dim=False,is_rgb=False,to_size=exp_config.original_image_size,val=True)
            # logging.error('SIDE OBJ ' + str(side_obj_softmaxed_v.shape))
            
            # save_array(side_obj_softmaxed_v,name_of_seq+'/side_obj_'+str(frame_number),slice_last_dim=True,is_rgb=False,to_size=exp_config.original_image_size,val=True)
            
            # logging.error('SAVED EXCEPT PROJ')
            
            # save_array(projected_obj_v,name_of_seq+'/projected_obj'+str(frame_number),slice_last_dim=True,is_rgb=False,to_size=exp_config.patch_size,val=True)
#            

            # save_array(projected_estimates_v,name_of_seq+'/projected_estimates'+str(frame_number),slice_last_dim=True,is_rgb=False,to_size=exp_config.patch_size,val=True)
#            
            # save_array(combined_projected_estimates_v,name_of_seq+'/combined_projected_estimates'+str(frame_number),slice_last_dim=True,is_rgb=False,to_size=exp_config.patch_size,val=True)
            
#                            save_array(mask2_v,'masks2',slice_last_dim=True,is_rgb=False)
            
            # save_array(occ_softmaxed_v,name_of_seq+'/occ_'+str(frame_number),slice_last_dim=True,is_rgb=False,to_size=exp_config.patch_size,val=True)
            # save_array(side_occ_masks_v,name_of_seq+'/side_occ'+str(frame_number),slice_last_dim=True,is_rgb=False,to_size=exp_config.original_image_size,val=True)

            scene_results.append(np.array(grid_j1s))
            # occ_scene_results.append(occ_all_stats)
            
            
            
        
        seq_j1 = np.array(scene_results)
        # if not (len(scene_results) > 0):
        #     continue
        # logging.error('SEQ J SHAPE ' + str(seq_j1.shape))
        all_j1s.append(np.squeeze(seq_j1))
     
        exists = seq_j1[...,6] 
        temp_res = seq_j1[...,2]/(seq_j1[...,2]+seq_j1[...,3]+seq_j1[...,4]+0.0001)
        
        framewise_best.append(np.argmax(temp_res,axis=0))
        # logging.error('ALL J1 ' + str(all_j1s))
        
        # seq_occ_j1 = np.array(occ_scene_results)
        # occ_all_j1s.append(np.squeeze(seq_occ_j1))
        temp_string = "Iteration : " + str(iteration) + " : Scene " + str(my_scene_token)+ " - j1: " + str(np.mean(temp_res,axis=0)) 
        # temp_string = " : Scene " + str(my_scene_token)+' best res : '+str(np.max(np.sum(temp_res,axis=-1)/np.sum(exists,axis=-1),axis=0)) + ' at ' +str(frame_numbers_list[np.argmax(np.sum(temp_res,axis=-1)/np.sum(exists,axis=-1),axis=0)]) 
        
        
        
        res_strings.append(temp_string)
        logging.error(temp_string)
        write_to_txt_file(os.path.join(log_dir,'val_results.txt'),[temp_string])
    
    # to_return = all_j1s
    
    # logging.error('ALL J1 ' + str(all_j1s))
    tot_j1 = np.concatenate(all_j1s,axis=0)
    # logging.error('TOT j '+str(tot_j1))
    # logging.error('TOT J ' + str(tot_j1.shape))
    # occ_tot_j1 = np.concatenate(occ_all_j1s,axis=0)

#    logging.error('BEV J ' )
#    logging.error(str(tot_j1))
#    
#    logging.error('IMAGE J ' )
#    logging.error(str(image_tot_j1))
#    
#    logging.error('TOT J SHAPE ' + str(image_tot_j1.shape))
    
    j = tot_j1[...,0]
    union = tot_j1[...,1]
    tp = tot_j1[...,2]
    fp = tot_j1[...,3]
    fn = tot_j1[...,4]
    tn = tot_j1[...,5]
    gt_exists = tot_j1[...,-1]
    
    
    
    
    # gt_exists_boolean =  > 0.5
    
    tp_rate = np.sum(tp,axis=0)/(np.sum(tp,axis=0) +  np.sum(fn,axis=0) + 0.0001)
    fp_rate = np.sum(fp,axis=0)/( np.sum(fp,axis=0) + np.sum(tn,axis=0) + 0.0001)
    
    # logging.error('TP RATE ' + str(np.mean(tp_rate,axis=(0,1))))
    # logging.error('FP RATE ' + str(np.mean(fp_rate,axis=(0,1))))
    
    tp_rate = np.sum(tp,axis=0)/(np.sum(tp,axis=0) +  np.sum(fn,axis=0) + 0.0001)
    fp_rate = np.sum(fp,axis=0)/( np.sum(fp,axis=0) + np.sum(tn,axis=0) + 0.0001)
    precision = np.sum(tp,axis=0)/(np.sum(tp,axis=0) +  np.sum(fp,axis=0) + 0.0001)
    
    # logging.error('TP RATE ' + str(np.mean(tp_rate,axis=(0,1))))
    # logging.error('FP RATE ' + str(np.mean(fp_rate,axis=(0,1))))
    
    take_all_j = np.mean(j,axis=0)
    confuse_iou = np.sum(tp,axis=0)/(np.sum(tp,axis=0) + np.sum(fp,axis=0) + np.sum(fn,axis=0) + 0.0001)
    
    temp_string = 'Bev framewise j : ' + str(take_all_j)+ ' , Bev confuse j : ' + str(confuse_iou) + '\n' +\
    ' Bev tp_rate : ' + str(tp_rate)+ ' Bev fp_rate : ' + str(fp_rate)+ ' Bev precision : ' + str(precision)+ '\n'
    
    
    logging.error(temp_string)
    res_strings.append(temp_string)
    write_to_txt_file(os.path.join(log_dir,'val_results.txt'),res_strings)
    return confuse_iou


def save_results_binary(query_image,prev_whole_label,query_gt,seq,frame, val_path):

       
   corrected_img =  query_image
#   logging.error('IMAGE SHAPE ' + str(corrected_img.shape))
   for k in range(len(prev_whole_label)):
       ann = np.squeeze(prev_whole_label[k])
       
#       logging.error('ANN SHAPE ' + str(ann.shape))
#       logging.error('QUERY IMG SHAPE ' + str(query_image.shape))
       img = overlay_semantic_mask(corrected_img, ann)
       img = img.astype(np.uint8)
       img_png=Image.fromarray(img)
       img_png = img_png.resize((800,450))
       img_png.save(os.path.join(val_path,'seq_'+str(seq)+'_frame_'+str(frame)+'_estimate'+str(k)+'.jpg'))         
       
   for k in range(query_gt.shape[-1]):
       ann = np.squeeze(query_gt[...,k])
       img = overlay_semantic_mask(corrected_img, ann)
       img = img.astype(np.uint8)
       img_png=Image.fromarray(img)
       img_png = img_png.resize((800,450))
       img_png.save(os.path.join(val_path,'seq_'+str(seq)+'_frame_'+str(frame)+'_labels'+str(k)+'.jpg'))
       
    
def expand_image(img,new_sizes,left_up):
    
    if len(img.shape) ==3:
    
        new_img = np.zeros((new_sizes[0],new_sizes[1],img.shape[2]),img.dtype)
        new_img[int(left_up[0]):int(left_up[0]+img.shape[0]),int(left_up[1]):int(left_up[1]+img.shape[1]),:] = img
    else:
        new_img = np.zeros((new_sizes[0],new_sizes[1]),img.dtype)
        new_img[int(left_up[0]):int(left_up[0]+img.shape[0]),int(left_up[1]):int(left_up[1]+img.shape[1])] = img
    
    
    return new_img


def standard_iterate_minibatches(my_scene,max_interval_between_frames,
                                                                          reference_frame_index,
                                                                         n_frames_per_seq=3,
                                                                         batch_size=1): 

    n_seqs = n_frames_per_seq-num_frames+1
    try:
        current_dir = os.path.join(target_dir,'scene'+my_scene)
        
#        logging.error('Cur directory : ' + str(current_dir))
        pool = ThreadPool(n_seqs*num_frames) 
        all_images = []
        all_labels = []
        small_labels=[]
        
        all_images_list = sorted(glob.glob(os.path.join(current_dir,'img*.png')))
        all_labels_list = sorted(glob.glob(os.path.join(current_dir,'label*.png')))
        
#        logging.error(str(all_images_list))
#        logging.error(str(os.listdir(current_dir)))
        
#        n_frames_in_scene = len(all_images_list)
#        frame_ids=[]
#        first_frame = random.randint(0,n_frames_in_scene-n_frames_per_seq)
#        frame_ids.append(first_frame)
        
        n_frames_in_scene = len(all_images_list)
        seq_length = n_frames_in_scene
        frame_ids=[]
        first_frame = random.randint(0,n_frames_in_scene-n_frames_per_seq)
        frame_ids.append(first_frame)
        
#        for frame_number in range(1,n_frames_per_seq):
#            frame_ids.append(random.randint(frame_ids[-1]+1, np.min([n_frames_in_scene-(n_frames_per_seq-frame_number),np.max([frame_ids[-1]+1,n_frames_in_scene-(n_frames_per_seq-frame_number-1)*2-1]),frame_ids[-1]+max_interval_between_frames])))
#            
        
        for frame_number in range(1,n_frames_per_seq):
            frame_ids.append(random.randint(frame_ids[-1]+1, np.min([seq_length-(n_frames_per_seq-frame_number),frame_ids[-1]+max_interval_between_frames])))
                
#        for frame_number in range(1,n_frames_per_seq):
#            frame_ids.append(random.randint(frame_ids[-1]+1, np.min([n_frames_in_scene-(n_frames_per_seq-frame_number),np.max([frame_ids[-1]+1,n_frames_in_scene-(n_frames_per_seq-frame_number-1)*2-1]),frame_ids[-1]+max_interval_between_frames])))
#            
#        for frame_number in range(1,n_frames_per_seq):
#            frame_ids.append(first_frame + frame_number)
            
        pairs = []
        
        
        scene_token = current_dir.split('/')[-1][5:]
        
        my_scene = nusc.get('scene', scene_token)
        
        first_sample_token = my_scene['first_sample_token']
        last_sample_token = my_scene['last_sample_token']
        first_sample_ind = nusc.getind('sample',first_sample_token)
        last_sample_ind = nusc.getind('sample',last_sample_token)
        
        all_sample_inds = np.arange(first_sample_ind,last_sample_ind+1)
        
        transforms_list=[]        
        reference_samples = []
        
#        logging.error('STARTING BEV TO BEV')
        
        bev_labels_list=[]
        
        for k in range(n_seqs):
      
                
            cur_ref_sample = nusc.sample[all_sample_inds[frame_ids[k+reference_frame_index]]]
            reference_samples.append(cur_ref_sample)
            
            cam_token_cur = cur_ref_sample['data']['CAM_FRONT']
            cam_record_cur = nusc.get('sample_data', cam_token_cur)
            
            bev_label = np.array(Image.open( os.path.join('/srv/beegfs02/scratch/tracezuerich/data/cany/monomaps_labels_vanilla',  
                                       cam_record_cur['token'] + '.png')),np.int32)
            
            bev_label = np.flipud(bev_label)
            
            bev_label = decode_binary_labels(bev_label,exp_config.num_bev_classes+1)

            bev_labels_list.append(bev_label)
            cs_record_cur = nusc.get('calibrated_sensor', cam_record_cur['calibrated_sensor_token'])
            cam_intrinsic = np.array(cs_record_cur['camera_intrinsic'])
            
            # np.savez('/home/cany/image_trans_stuff.npy',cs_record_cur,cam_intrinsic)
            
            to_image_transform = project_to_image(np.zeros((exp_config.project_base_patch_size[1],exp_config.project_base_patch_size[0])),cs_record_cur,cam_intrinsic)

#        logging.error('BEV TO BEV ENDED')
        

        
#        my_sample = nusc.sample[scene_samples[m]]
        for k in range(n_seqs):
            for m in range(num_frames):
                pairs.append((all_images_list[frame_ids[k + m]],all_labels_list[frame_ids[k+m]],reference_samples[k],nusc.sample[all_sample_inds[frame_ids[k+m]]],m==reference_frame_index))
        

        results = pool.map(single_process,pairs)
        
        
        pool.close() 
        pool.join() 
#        logging.error('Results shape : ' + str(len(results)))

        seq_images_ar=np.zeros((n_frames_per_seq,exp_config.camera_image_patch_size[0],exp_config.camera_image_patch_size[1],3),np.float32)
        seq_labels_ar=np.ones((n_frames_per_seq,int(exp_config.camera_image_patch_size[0]/4),int(exp_config.camera_image_patch_size[1]/4),int(total_label_slices  )),np.float32)

        bev_transforms_ar1=np.ones((n_seqs,num_frames,8),np.float32)
        bev_transforms_ar2=np.ones((n_seqs,num_frames,8),np.float32)
        bev_transforms_ar3=np.ones((n_seqs,num_frames,8),np.float32)
        coordinate_transforms_ar=np.ones((n_seqs,num_frames,3,3),np.float32)
        bev_images_ar=np.zeros((n_seqs,num_frames,exp_config.patch_size[1],exp_config.patch_size[0],3),np.float32)
        bev_labels_ar=np.ones((n_seqs,num_frames,exp_config.patch_size[1],exp_config.patch_size[0],int(total_label_slices  )),np.float32)
        bev_covers_ar=np.ones((n_seqs,num_frames,exp_config.patch_size[1],exp_config.patch_size[0],1),np.float32)

#        logging.error('PROJECT TO GROUND ENDED')
        for k in range(len(results)):
            temp_res = results[k]
            
            if k < num_frames:
            
                seq_images_ar[k,...] = np.copy(temp_res[0])
                seq_labels_ar[k,...] = np.copy(temp_res[1])
           
            elif k >= (n_seqs*num_frames - (n_seqs - 1)):
                seq_images_ar[k - (num_frames-1),...] = np.copy(temp_res[0])
                seq_labels_ar[k - (num_frames-1),...] = np.copy(temp_res[1])
            
#                logging.error('RETURNED GRID SHAPE ' + str(temp_res[3].shape))
            bev_images_ar[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[2])
            bev_labels_ar[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[3])
            bev_covers_ar[int(k//num_frames),k%num_frames,...] = np.expand_dims(np.copy(temp_res[4][...,0]),axis=-1)
            
            bev_transforms_ar1[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[6])
            bev_transforms_ar2[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[7])
            bev_transforms_ar3[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[8])
            coordinate_transforms_ar[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[5])
   
        
        # if n_seqs > 1:
        
        #     return seq_images_ar, seq_labels_ar,bev_images_ar,bev_labels_ar,bev_covers_ar,np.stack(transforms_list,axis=0), bev_transforms_ar1,bev_transforms_ar2,bev_transforms_ar3,coordinate_transforms_ar,np.stack(bev_labels_list,axis=0),True
        # else:
        return seq_images_ar, seq_labels_ar,bev_images_ar,bev_labels_ar,bev_covers_ar,np.zeros((1,8)), bev_transforms_ar1,bev_transforms_ar2,bev_transforms_ar3,coordinate_transforms_ar,np.stack(bev_labels_list,axis=0),to_image_transform,True

    except Exception as e:
        pool.close() 
        pool.join() 
        logging.error('Exception ' + str(e))
            
        return None,None,None,None,None, None,None,None,None,None,None,None,False





def tensorflow_project_to_ground(image1,source_image1,pose1, cs1,pose2,cs2, cam_intrinsic,reference_frame=False,grid=False):
    
    extents = exp_config.project_extents
    resolution = exp_config.project_resolution
        
    extra_space = exp_config.project_extra_space
    
    sample_point1 = np.array([int(3*image1.shape[1]/8),int(7*image1.shape[0]/8),1],np.float32)
    sample_point2 = np.array([int(5*image1.shape[1]/8),int(7*image1.shape[0]/8),1],np.float32)
    sample_point3 = np.array([int(3*image1.shape[1]/8),int(6*image1.shape[0]/8),1],np.float32)
    sample_point4 = np.array([int(5*image1.shape[1]/8),int(6*image1.shape[0]/8),1],np.float32)
    
    base_sample_points = np.stack([sample_point1,sample_point2,sample_point3,sample_point4],axis=-1)
    
    source_sample_point1 = np.array([int(3*source_image1.shape[1]/8),int(7*source_image1.shape[0]/8),1],np.float32)
    source_sample_point2 = np.array([int(5*source_image1.shape[1]/8),int(7*source_image1.shape[0]/8),1],np.float32)
    source_sample_point3 = np.array([int(3*source_image1.shape[1]/8),int(6*source_image1.shape[0]/8),1],np.float32)
    source_sample_point4 = np.array([int(5*source_image1.shape[1]/8),int(6*source_image1.shape[0]/8),1],np.float32)
    
    source_base_sample_points = np.stack([source_sample_point1,source_sample_point2,source_sample_point3,source_sample_point4],axis=-1)
    
    source_points = np.float32(source_base_sample_points[0:2,:].T)
    
    inverse_intrinsic =  np.linalg.inv(cam_intrinsic)
    
    pr = Quaternion(pose1['rotation']).rotation_matrix.T
    pr2 = Quaternion(pose2['rotation']).rotation_matrix.T
    
    inverse_pr = np.linalg.inv(pr)
    inverse_pr2 = np.linalg.inv(pr2)
    pt = np.array(pose1['translation']).reshape((-1, 1))
    pt2 = np.array(pose2['translation']).reshape((-1, 1))
    
    cr = Quaternion(cs1['rotation']).rotation_matrix.T
    cr2 = Quaternion(cs2['rotation']).rotation_matrix.T
    
    inverse_cr = np.linalg.inv(cr)
    inverse_cr2 = np.linalg.inv(cr2)
    
    ct = np.array(cs1['translation']).reshape((-1, 1))
    ct2 = np.array(cs2['translation']).reshape((-1, 1))
    
#    coef_matrix = np.dot(inverse_cr,inverse_intrinsic)
    coef_matrix = np.dot(inverse_cr,inverse_intrinsic)
    trans1 = ct
    
    
    trans2 = pt 
    if reference_frame:
        sample_points = base_sample_points*cam_intrinsic[0,0]*np.array(cs1['translation']).reshape((-1, 1))[-1]/(np.abs(base_sample_points[1,:]-cam_intrinsic[1,-1]))
        
        
        
        prenorm_corresponding_points = np.dot(coef_matrix,sample_points) + trans1 
        prenorm_corresponding_points[2,:] = prenorm_corresponding_points[0,:]
        prenorm_corresponding_points[0,:] = -prenorm_corresponding_points[1,:]
        prenorm_corresponding_points[1,:] = prenorm_corresponding_points[2,:]
        corresponding_points = prenorm_corresponding_points[0:2,:]
    #    corresponding_points[2,:] = 0.0
        
     
        corresponding_points = np.float32(corresponding_points.T)
        
        corresponding_points[:,0] = (corresponding_points[:,0] + extents[1])/resolution
        my_ys = corresponding_points[:,1]
        my_ys = my_ys - extents[2]
        my_ys = (extents[3] - extents[2])/resolution - my_ys/resolution
        corresponding_points[:,1] = my_ys
        
        corresponding_points[:,0] = corresponding_points[:,0] + extra_space[0]/2
        corresponding_points[:,1] = corresponding_points[:,1] + extra_space[1]/2
        
        
#        
        estimated_transform = cv2.getPerspectiveTransform(corresponding_points,source_points)
        
#        warped_image_ref = cv2.warpPerspective(image1,modified_transform,(400,400))
        
        return estimated_transform
    
        '''
        HANDLE SECOND IMAGE
        '''
    
    else:
        sample_points = base_sample_points*cam_intrinsic[0,0]*np.array(cs2['translation']).reshape((-1, 1))[-1]/(np.abs(base_sample_points[1,:]-cam_intrinsic[1,-1]))
        
        coef_matrix = np.dot(inverse_pr2,np.dot(inverse_cr,inverse_intrinsic))
        
        trans1 = np.dot(inverse_pr2,ct2)
        
        trans2 = pt2 - pt 
        
        prenorm_corresponding_points = np.dot(pr,np.dot(coef_matrix,sample_points) + trans1 + trans2)
        prenorm_corresponding_points[2,:] = prenorm_corresponding_points[0,:]
        prenorm_corresponding_points[0,:] = -prenorm_corresponding_points[1,:]
        prenorm_corresponding_points[1,:] = prenorm_corresponding_points[2,:]
        corresponding_points = prenorm_corresponding_points[0:2,:]
    #    corresponding_points[2,:] = 0.0
        
  
         
        corresponding_points = np.float32(corresponding_points.T)
        
        corresponding_points[:,0] = (corresponding_points[:,0] + extents[1])/resolution
        my_ys = corresponding_points[:,1]
        my_ys = my_ys - extents[2]
        my_ys = (extents[3] - extents[2])/resolution - my_ys/resolution
        corresponding_points[:,1] = my_ys
        
        corresponding_points[:,0] = corresponding_points[:,0] + extra_space[0]/2
        corresponding_points[:,1] = corresponding_points[:,1] + extra_space[1]/2
        
        
#        
        estimated_transform = cv2.getPerspectiveTransform(corresponding_points,source_points)
#        

    
        return estimated_transform
    
def project_to_ground(image1,label1,pose1, cs1,pose2,cs2, cam_intrinsic,vis_mask,reference_frame=False,grid=False):
    
    image1 = image1*vis_mask
#    image2 = image2*vis_mask

    covered_region = np.ones_like(image1) * vis_mask
    
    sample_point1 = np.array([int(3*image1.shape[1]/8),int(7*image1.shape[0]/8),1],np.float32)
    sample_point2 = np.array([int(5*image1.shape[1]/8),int(7*image1.shape[0]/8),1],np.float32)
    sample_point3 = np.array([int(3*image1.shape[1]/8),int(6*image1.shape[0]/8),1],np.float32)
    sample_point4 = np.array([int(5*image1.shape[1]/8),int(6*image1.shape[0]/8),1],np.float32)
    
    base_sample_points = np.stack([sample_point1,sample_point2,sample_point3,sample_point4],axis=-1)
    
    source_points = np.float32(base_sample_points[0:2,:].T)
    
    inverse_intrinsic =  np.linalg.inv(cam_intrinsic)
    
    pr = Quaternion(pose1['rotation']).rotation_matrix.T
    pr2 = Quaternion(pose2['rotation']).rotation_matrix.T
    
    inverse_pr = np.linalg.inv(pr)
    inverse_pr2 = np.linalg.inv(pr2)
    pt = np.array(pose1['translation']).reshape((-1, 1))
    pt2 = np.array(pose2['translation']).reshape((-1, 1))
    
    cr = Quaternion(cs1['rotation']).rotation_matrix.T
    cr2 = Quaternion(cs2['rotation']).rotation_matrix.T
    
    inverse_cr = np.linalg.inv(cr)
    inverse_cr2 = np.linalg.inv(cr2)
    
    ct = np.array(cs1['translation']).reshape((-1, 1))
    ct2 = np.array(cs2['translation']).reshape((-1, 1))
    
#    coef_matrix = np.dot(inverse_cr,inverse_intrinsic)
    coef_matrix = np.dot(inverse_cr,inverse_intrinsic)
    trans1 = ct
    
    
    trans2 = pt 
    
    total_image_size = exp_config.total_image_size
    
    if reference_frame:
        sample_points = base_sample_points*cam_intrinsic[0,0]*np.array(cs1['translation']).reshape((-1, 1))[-1]/(np.abs(base_sample_points[1,:]-cam_intrinsic[1,-1]))
        
        
        
        prenorm_corresponding_points = np.dot(coef_matrix,sample_points) + trans1 
        prenorm_corresponding_points[2,:] = prenorm_corresponding_points[0,:]
        prenorm_corresponding_points[0,:] = -prenorm_corresponding_points[1,:]
        prenorm_corresponding_points[1,:] = prenorm_corresponding_points[2,:]
        corresponding_points = prenorm_corresponding_points[0:2,:]
    #    corresponding_points[2,:] = 0.0
        
        extents = exp_config.extents
        resolution = exp_config.resolution
        
         
        corresponding_points = np.float32(corresponding_points.T)
        
        corresponding_points[:,0] = (corresponding_points[:,0] + extents[1])/resolution
        my_ys = corresponding_points[:,1]
        my_ys = my_ys - extents[2]
        my_ys = (extents[3] - extents[2])/resolution - my_ys/resolution
        corresponding_points[:,1] = my_ys
        
        estimated_transform = cv2.getPerspectiveTransform(source_points,corresponding_points)
        
        modified_transform = np.copy(estimated_transform)
    #        
        translation_matrix = np.eye(3)
        
        extra_space = exp_config.extra_space
        
        translation_matrix[0,-1] = extra_space[0]/2
        translation_matrix[1,-1] = extra_space[1]/2
        
        modified_transform = np.dot(translation_matrix, modified_transform)
        
#        logging.error('PROJECT TO GROUND REF TRANS FOUND')
        
        warped_image_ref = cv2.warpPerspective(image1,modified_transform,total_image_size,flags=cv2.INTER_LINEAR)
        warped_covered = cv2.warpPerspective(covered_region,modified_transform,total_image_size,flags=cv2.INTER_LINEAR)
        warped_label = cv2.warpPerspective(label1,modified_transform,total_image_size,flags=cv2.INTER_NEAREST)
        
        
#        logging.error('PROJECT TO GROUND REF WARPED')
        
        return warped_image_ref, warped_covered, warped_label, modified_transform
    
        '''
        HANDLE SECOND IMAGE
        '''
    
    else:
        sample_points = base_sample_points*cam_intrinsic[0,0]*np.array(cs2['translation']).reshape((-1, 1))[-1]/(np.abs(base_sample_points[1,:]-cam_intrinsic[1,-1]))
        
        coef_matrix = np.dot(inverse_pr2,np.dot(inverse_cr,inverse_intrinsic))
        
        trans1 = np.dot(inverse_pr2,ct2)
        
        trans2 = pt2 - pt 
        
        prenorm_corresponding_points = np.dot(pr,np.dot(coef_matrix,sample_points) + trans1 + trans2)
        prenorm_corresponding_points[2,:] = prenorm_corresponding_points[0,:]
        prenorm_corresponding_points[0,:] = -prenorm_corresponding_points[1,:]
        prenorm_corresponding_points[1,:] = prenorm_corresponding_points[2,:]
        corresponding_points = prenorm_corresponding_points[0:2,:]
    #    corresponding_points[2,:] = 0.0
        
        extents = exp_config.extents
        resolution = exp_config.resolution
        
         
        corresponding_points = np.float32(corresponding_points.T)
        
        corresponding_points[:,0] = (corresponding_points[:,0] + extents[1])/resolution
        my_ys = corresponding_points[:,1]
        my_ys = my_ys - extents[2]
        my_ys = (extents[3] - extents[2])/resolution - my_ys/resolution
        corresponding_points[:,1] = my_ys
        
        estimated_transform = cv2.getPerspectiveTransform(source_points,corresponding_points)
        
        modified_transform = np.copy(estimated_transform)
    #        
        translation_matrix = np.eye(3)
        
        
        extra_space = exp_config.extra_space
        
        translation_matrix[0,-1] = extra_space[0]/2
        translation_matrix[1,-1] = extra_space[1]/2
        
        modified_transform = np.dot(translation_matrix, modified_transform)
        
        
#        logging.error('PROJECT TO GROUND DEST TRANS FOUND')
        
        
        warped_image_dest = cv2.warpPerspective(image1,modified_transform,total_image_size,flags=cv2.INTER_LINEAR)
        warped_covered = cv2.warpPerspective(covered_region,modified_transform,total_image_size,flags=cv2.INTER_LINEAR)
        warped_label = cv2.warpPerspective(label1,modified_transform,total_image_size,flags=cv2.INTER_NEAREST)
        
        
#        logging.error('PROJECT TO GROUND DEST WARPED')
        
        return warped_image_dest,  warped_covered, warped_label, modified_transform

def tensorflow_project_bev_to_bev(cur_sample, next_sample):
    
    camera_channel='CAM_FRONT'
    cam_token_ref = next_sample['data'][camera_channel]
    cam_record_ref = nusc.get('sample_data', cam_token_ref)
    
    cs1 = nusc.get('calibrated_sensor', cam_record_ref['calibrated_sensor_token'])
    cam_intrinsic = np.array(cs1['camera_intrinsic'])

    pose1 = nusc.get('ego_pose', cam_record_ref['ego_pose_token'])
    
    '''
    '''
    cam_token_cur = cur_sample['data'][camera_channel]
    cam_record_cur = nusc.get('sample_data', cam_token_cur)
    
    cs2 = nusc.get('calibrated_sensor', cam_record_cur['calibrated_sensor_token'])
   
    pose2 = nusc.get('ego_pose', cam_record_cur['ego_pose_token'])
    
    '''
    MAPS BEV IMAGE2 to BEV IMAGE1
    '''
    
    extents = exp_config.extents
    resolution = exp_config.resolution
    extra_space = exp_config.extra_space
    
    image1 = np.zeros((int(exp_config.total_image_size[1]/exp_config.feature_downsample),int(exp_config.total_image_size[0]/exp_config.feature_downsample)))
    
    sample_point1 = np.array([int(3*image1.shape[1]/8),int(3*image1.shape[0]/8),1],np.float32)
    sample_point2 = np.array([int(5*image1.shape[1]/8),int(3*image1.shape[0]/8),1],np.float32)
    sample_point3 = np.array([int(3*image1.shape[1]/8),int(5*image1.shape[0]/8),1],np.float32)
    sample_point4 = np.array([int(5*image1.shape[1]/8),int(5*image1.shape[0]/8),1],np.float32)
    
    base_sample_points = np.stack([sample_point1,sample_point2,sample_point3,sample_point4],axis=-1)
    source_points = np.copy(np.float32(base_sample_points[0:2,:].T))
    
    base_sample_points[0:2,:] = base_sample_points[0:2,:] - extra_space[0]/2/exp_config.feature_downsample
    
    my_x = base_sample_points[0,:]
    my_y = base_sample_points[1,:]
    
    my_x = my_x*resolution*exp_config.feature_downsample + extents[0]
    my_y = ((extents[3] - extents[2])/resolution/exp_config.feature_downsample - my_y)*resolution*exp_config.feature_downsample + extents[2]
    
    real_sample_points = np.copy(np.stack([my_x,my_y,np.ones_like(my_x)],axis=0))
    
    inverse_intrinsic =  np.linalg.inv(cam_intrinsic)
    
    pr = Quaternion(pose1['rotation']).rotation_matrix.T
    pr2 = Quaternion(pose2['rotation']).rotation_matrix.T
    
    inverse_pr = np.linalg.inv(pr)
    inverse_pr2 = np.linalg.inv(pr2)
    pt = np.array(pose1['translation']).reshape((-1, 1))
    pt2 = np.array(pose2['translation']).reshape((-1, 1))
    
    cr = Quaternion(cs1['rotation']).rotation_matrix.T
    cr2 = Quaternion(cs2['rotation']).rotation_matrix.T
    
    inverse_cr = np.linalg.inv(cr)
    inverse_cr2 = np.linalg.inv(cr2)
    
    ct = np.array(cs1['translation']).reshape((-1, 1))
    ct2 = np.array(cs2['translation']).reshape((-1, 1))
   
    
    sample_points = base_sample_points
    
    coef_matrix = inverse_pr2
    trans1 = np.dot(inverse_pr2,ct2)
    trans2 = pt2 - pt 
    
    
    
    real_sample_points[2,:] = real_sample_points[0,:]
    real_sample_points[0,:] = real_sample_points[1,:]
    real_sample_points[1,:] = -real_sample_points[2,:]
    
    real_sample_points[2,:] = 1
    
    prenorm_corresponding_points = np.dot(pr,np.dot(coef_matrix,real_sample_points)  + trans2)
    
#    norm_const = np.copy(prenorm_corresponding_points[2,:])
    
    prenorm_corresponding_points[2,:] = prenorm_corresponding_points[0,:]
    prenorm_corresponding_points[0,:] = -prenorm_corresponding_points[1,:]
    prenorm_corresponding_points[1,:] = prenorm_corresponding_points[2,:]
    corresponding_points = prenorm_corresponding_points[0:2,:]
#    corresponding_points[2,:] = 0.0
    
    corresponding_points = np.float32(corresponding_points.T)
    
    corresponding_points[:,0] = (corresponding_points[:,0] + extents[1])/resolution/exp_config.feature_downsample
    my_ys = corresponding_points[:,1]
    my_ys = my_ys - extents[2]
    my_ys = (extents[3] - extents[2])/resolution/exp_config.feature_downsample - my_ys/resolution/exp_config.feature_downsample
    corresponding_points[:,1] = my_ys
    
        
    corresponding_points[:,0] = corresponding_points[:,0] + extra_space[0]/2/exp_config.feature_downsample
    corresponding_points[:,1] = corresponding_points[:,1] + extra_space[1]/2/exp_config.feature_downsample
    
    estimated_transform = cv2.getPerspectiveTransform(corresponding_points,source_points)


    return estimated_transform
    
def project_bev_to_bev(cur_sample, next_sample, cur_label, next_label):
    
    camera_channel='CAM_FRONT'
    cam_token_ref = next_sample['data'][camera_channel]
    cam_record_ref = nusc.get('sample_data', cam_token_ref)
    
    cs1 = nusc.get('calibrated_sensor', cam_record_ref['calibrated_sensor_token'])
    cam_intrinsic = np.array(cs1['camera_intrinsic'])

    pose1 = nusc.get('ego_pose', cam_record_ref['ego_pose_token'])
    
    '''
    '''
    cam_token_cur = cur_sample['data'][camera_channel]
    cam_record_cur = nusc.get('sample_data', cam_token_cur)
    
    cs2 = nusc.get('calibrated_sensor', cam_record_cur['calibrated_sensor_token'])
   
    pose2 = nusc.get('ego_pose', cam_record_cur['ego_pose_token'])
    
    '''
    MAPS BEV IMAGE2 to BEV IMAGE1
    '''
    
    extents = exp_config.extents
    resolution = exp_config.resolution
    extra_space = exp_config.extra_space
    
#    image1 = np.zeros((int(exp_config.total_image_size[1]/exp_config.feature_downsample),int(exp_config.total_image_size[0]/exp_config.feature_downsample)))
#   
    image1 = np.zeros_like(cur_label)
    
    sample_point1 = np.array([int(3*image1.shape[1]/8),int(3*image1.shape[0]/8),1],np.float32)
    sample_point2 = np.array([int(5*image1.shape[1]/8),int(3*image1.shape[0]/8),1],np.float32)
    sample_point3 = np.array([int(3*image1.shape[1]/8),int(5*image1.shape[0]/8),1],np.float32)
    sample_point4 = np.array([int(5*image1.shape[1]/8),int(5*image1.shape[0]/8),1],np.float32)
    
    base_sample_points = np.stack([sample_point1,sample_point2,sample_point3,sample_point4],axis=-1)
    source_points = np.copy(np.float32(base_sample_points[0:2,:].T))
    
    base_sample_points[0:2,:] = base_sample_points[0:2,:] - extra_space[0]/2
    
    my_x = base_sample_points[0,:]
    my_y = base_sample_points[1,:]
    
    my_x = my_x*resolution*exp_config.feature_downsample + extents[0]
    my_y = ((extents[3] - extents[2])/resolution/exp_config.feature_downsample - my_y)*resolution*exp_config.feature_downsample + extents[2]
    
    real_sample_points = np.copy(np.stack([my_x,my_y,np.ones_like(my_x)],axis=0))
    
    inverse_intrinsic =  np.linalg.inv(cam_intrinsic)
    
    pr = Quaternion(pose1['rotation']).rotation_matrix.T
    pr2 = Quaternion(pose2['rotation']).rotation_matrix.T
    
    inverse_pr = np.linalg.inv(pr)
    inverse_pr2 = np.linalg.inv(pr2)
    pt = np.array(pose1['translation']).reshape((-1, 1))
    pt2 = np.array(pose2['translation']).reshape((-1, 1))
    
    cr = Quaternion(cs1['rotation']).rotation_matrix.T
    cr2 = Quaternion(cs2['rotation']).rotation_matrix.T
    
    inverse_cr = np.linalg.inv(cr)
    inverse_cr2 = np.linalg.inv(cr2)
    
    ct = np.array(cs1['translation']).reshape((-1, 1))
    ct2 = np.array(cs2['translation']).reshape((-1, 1))
   
    
    sample_points = base_sample_points
    
    coef_matrix = inverse_pr2
    trans1 = np.dot(inverse_pr2,ct2)
    trans2 = pt2 - pt 
    
    
    
    real_sample_points[2,:] = real_sample_points[0,:]
    real_sample_points[0,:] = real_sample_points[1,:]
    real_sample_points[1,:] = -real_sample_points[2,:]
    
    real_sample_points[2,:] = 1
    
    prenorm_corresponding_points = np.dot(pr,np.dot(coef_matrix,real_sample_points)  + trans2)
    
#    norm_const = np.copy(prenorm_corresponding_points[2,:])
    
    prenorm_corresponding_points[2,:] = prenorm_corresponding_points[0,:]
    prenorm_corresponding_points[0,:] = -prenorm_corresponding_points[1,:]
    prenorm_corresponding_points[1,:] = prenorm_corresponding_points[2,:]
    corresponding_points = prenorm_corresponding_points[0:2,:]
#    corresponding_points[2,:] = 0.0
    
    corresponding_points = np.float32(corresponding_points.T)
    
    corresponding_points[:,0] = (corresponding_points[:,0] + extents[1])/resolution/exp_config.feature_downsample
    my_ys = corresponding_points[:,1]
    my_ys = my_ys - extents[2]
    my_ys = (extents[3] - extents[2])/resolution/exp_config.feature_downsample - my_ys/resolution/exp_config.feature_downsample
    corresponding_points[:,1] = my_ys
    
        
    corresponding_points[:,0] = corresponding_points[:,0] + extra_space[0]/2/exp_config.feature_downsample
    corresponding_points[:,1] = corresponding_points[:,1] + extra_space[1]/2/exp_config.feature_downsample
    
    estimated_transform = cv2.getPerspectiveTransform(corresponding_points,source_points)


    return estimated_transform

#   
#
#def project_bev_to_bev(cur_sample, next_sample):
#    
#    camera_channel='CAM_FRONT'
#    cam_token_ref = next_sample['data'][camera_channel]
#    cam_record_ref = nusc.get('sample_data', cam_token_ref)
#    
#    cs1 = nusc.get('calibrated_sensor', cam_record_ref['calibrated_sensor_token'])
#    cam_intrinsic = np.array(cs1['camera_intrinsic'])
#
#    pose1 = nusc.get('ego_pose', cam_record_ref['ego_pose_token'])
#    
#    '''
#    '''
#    cam_token_cur = cur_sample['data'][camera_channel]
#    cam_record_cur = nusc.get('sample_data', cam_token_cur)
#    
#    cs2 = nusc.get('calibrated_sensor', cam_record_cur['calibrated_sensor_token'])
#   
#    pose2 = nusc.get('ego_pose', cam_record_cur['ego_pose_token'])
#    
#    '''
#    MAPS BEV IMAGE2 to BEV IMAGE1
#    '''
#    
#    extents = exp_config.extents
#    resolution = exp_config.resolution
#    extra_space = exp_config.extra_space
#    
#    image1 = np.zeros((exp_config.total_image_size[1],exp_config.total_image_size[0]))
#    
#    sample_point1 = np.array([int(3*image1.shape[1]/8),int(3*image1.shape[0]/8),1],np.float32)
#    sample_point2 = np.array([int(5*image1.shape[1]/8),int(3*image1.shape[0]/8),1],np.float32)
#    sample_point3 = np.array([int(3*image1.shape[1]/8),int(5*image1.shape[0]/8),1],np.float32)
#    sample_point4 = np.array([int(5*image1.shape[1]/8),int(5*image1.shape[0]/8),1],np.float32)
#    
#    base_sample_points = np.stack([sample_point1,sample_point2,sample_point3,sample_point4],axis=-1)
#    source_points = np.copy(np.float32(base_sample_points[0:2,:].T))
#    
#    base_sample_points[0:2,:] = base_sample_points[0:2,:] - extra_space[0]/2
#    
#    my_x = base_sample_points[0,:]
#    my_y = base_sample_points[1,:]
#    
#    my_x = my_x*resolution + extents[0]
#    my_y = ((extents[3] - extents[2])/resolution - my_y)*resolution + extents[2]
#    
#    real_sample_points = np.copy(np.stack([my_x,my_y,np.ones_like(my_x)],axis=0))
#    
#    inverse_intrinsic =  np.linalg.inv(cam_intrinsic)
#    
#    pr = Quaternion(pose1['rotation']).rotation_matrix.T
#    pr2 = Quaternion(pose2['rotation']).rotation_matrix.T
#    
#    inverse_pr = np.linalg.inv(pr)
#    inverse_pr2 = np.linalg.inv(pr2)
#    pt = np.array(pose1['translation']).reshape((-1, 1))
#    pt2 = np.array(pose2['translation']).reshape((-1, 1))
#    
#    cr = Quaternion(cs1['rotation']).rotation_matrix.T
#    cr2 = Quaternion(cs2['rotation']).rotation_matrix.T
#    
#    inverse_cr = np.linalg.inv(cr)
#    inverse_cr2 = np.linalg.inv(cr2)
#    
#    ct = np.array(cs1['translation']).reshape((-1, 1))
#    ct2 = np.array(cs2['translation']).reshape((-1, 1))
#   
#    
#    sample_points = base_sample_points
#    
#    coef_matrix = inverse_pr2
#    trans1 = np.dot(inverse_pr2,ct2)
#    trans2 = pt2 - pt 
#    
#    
#    
#    real_sample_points[2,:] = real_sample_points[0,:]
#    real_sample_points[0,:] = real_sample_points[1,:]
#    real_sample_points[1,:] = -real_sample_points[2,:]
#    
#    real_sample_points[2,:] = 1
#    
#    prenorm_corresponding_points = np.dot(pr,np.dot(coef_matrix,real_sample_points)  + trans2)
#    
##    norm_const = np.copy(prenorm_corresponding_points[2,:])
#    
#    prenorm_corresponding_points[2,:] = prenorm_corresponding_points[0,:]
#    prenorm_corresponding_points[0,:] = -prenorm_corresponding_points[1,:]
#    prenorm_corresponding_points[1,:] = prenorm_corresponding_points[2,:]
#    corresponding_points = prenorm_corresponding_points[0:2,:]
##    corresponding_points[2,:] = 0.0
#    
#    corresponding_points = np.float32(corresponding_points.T)
#    
#    corresponding_points[:,0] = (corresponding_points[:,0] + extents[1])/resolution
#    my_ys = corresponding_points[:,1]
#    my_ys = my_ys - extents[2]
#    my_ys = (extents[3] - extents[2])/resolution - my_ys/resolution
#    corresponding_points[:,1] = my_ys
#    
#        
#    corresponding_points[:,0] = corresponding_points[:,0] + extra_space[0]/2
#    corresponding_points[:,1] = corresponding_points[:,1] + extra_space[1]/2
#    
#    estimated_transform = cv2.getPerspectiveTransform(corresponding_points,source_points)
#
#
#    return estimated_transform
    
def iterate_minibatches(train_tokens,max_interval_between_frames, reference_frame_index, n_frames_per_seq=3,
                                                                         batch_size=BATCH_SIZE): 
    '''
    Function to create mini batches from the dataset of a certain batch size 
    Returns tuple of   ( t x n x h x w x 3, t x n x h x w x 1)  where t is  n_frames_per_seq and n is batch size
    This way it is easier to slice per frame number    
    '''

    n_videos = len(train_tokens)
  
    

    
    for b_i in range(0, n_videos, batch_size):

        if b_i + batch_size > n_videos:
            continue
         
        """
        Which video sequences will be in this batch
        """
        
#        batch_indices = np.sort(random_indices[b_i:b_i+batch_size])

        
        seq_images_ar, seq_labels_ar, bev_images_ar,bev_labels_ar,bev_covers_ar, transforms_ar,tf_transforms1,tf_transforms2,tf_transforms3, coordinate_transforms_ar,real_ref_bev_labels,to_image_transform,went_well =  standard_iterate_minibatches(train_tokens[b_i],max_interval_between_frames,
                                                                         
                                                                         reference_frame_index,
                                                                         n_frames_per_seq=n_frames_per_seq,
                                                                         batch_size=batch_size
                                                                         )
        
        
        if not went_well:
            continue
        squeezed_bev_covers_ar = np.squeeze(bev_covers_ar,axis=-1)
        
        total_img_list=[]
        total_labels_list=[]
    
        for k in range(n_seqs):
            total_img = np.zeros_like(bev_images_ar[0,0,...])
            total_labels = np.zeros_like(bev_labels_ar[0,0,...])
            for m in range(num_frames):
                total_img[squeezed_bev_covers_ar[k,m,...]>0.5,:] = bev_images_ar[k,m,...][squeezed_bev_covers_ar[k,m,...]>0.5,:]
                total_labels[squeezed_bev_covers_ar[k,m,...]>0.5,:] = bev_labels_ar[k,m,...][squeezed_bev_covers_ar[k,m,...]>0.5,:]
                
            total_img_list.append(total_img)
            total_labels_list.append(total_labels)
        
        fin_bev_images = np.stack(total_img_list,axis=0)
        fin_bev_labels = np.stack(total_labels_list,axis=0)
        fin_covers = np.clip(np.sum(bev_covers_ar,axis=1),0,1)
        
        my_area = np.float32(bev_covers_ar > 0.5)
        
#        logging.error('BEV IMAGES MAX ' + str(np.max(bev_images_ar))+ ' MIN ' + str(np.min(bev_images_ar)))
        
        separate_bev_images = my_area*bev_images_ar
        
        to_return_bev_images_list = []
        
        for k in range(num_frames):
            to_return_bev_images_list.append(separate_bev_images[:,k,...])
            
        to_return_bev_images = np.concatenate(to_return_bev_images_list,axis=-1)
        
        image_objects = cv2.warpPerspective(np.squeeze(real_ref_bev_labels[...,4:-1]),to_image_transform,exp_config.original_image_size,flags=cv2.INTER_LINEAR)
        image_objects= cv2.resize(image_objects,(int(exp_config.camera_image_patch_size[1]/4),int(exp_config.camera_image_patch_size[0]/4)), interpolation = cv2.INTER_LINEAR)
        image_objects = np.expand_dims(np.float32(image_objects > 0.5),axis=0)
        
        image_objects = np.concatenate([image_objects,np.clip(1-np.sum(image_objects,axis=-1,keepdims=True),0,1)],axis=-1)
        
        yield seq_images_ar, seq_labels_ar, fin_bev_images, fin_bev_labels,fin_covers , transforms_ar,tf_transforms1,tf_transforms2,tf_transforms3, bev_covers_ar, coordinate_transforms_ar,to_return_bev_images,real_ref_bev_labels,image_objects


def get_all_stats(annotation, segmentation,tot_mask,mask_iou=True):
    
    void_pixels = 1-tot_mask
    
#    annotation = annotation.astype(np.bool)& void_pixels
#    segmentation = segmentation.astype(np.bool)& void_pixels
    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)
    
    tp = np.float32(np.sum((segmentation & annotation) & void_pixels))
    fp = np.float32(np.sum((segmentation & np.logical_not(annotation)) & void_pixels))
          
    fn = np.float32(np.sum((np.logical_not(segmentation) & annotation) & void_pixels))
    tn = np.float32(np.sum((np.logical_not(segmentation) & np.logical_not(annotation)) & void_pixels))
    
    gt_exists = np.float32(np.sum(annotation & void_pixels) > 0)
    
    
    inters = np.sum((segmentation & annotation) & void_pixels)
    union = np.sum((segmentation | annotation) & void_pixels)

    j = inters / union
    
#    logging.error('J '  + str(j))
#    logging.error('UNION '  + str(np.isclose(union, 0)))
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    
    j = np.float32(j)
    
#    logging.error('J '  + str(j))
    union = np.float32(union)
    return np.array([j,union,tp,fp,fn,tn,gt_exists]), void_pixels


def get_confusion(annotation, segmentation,tot_mask,mask_iou=True):
    
    void_pixels = 1-tot_mask
    
#    annotation = annotation.astype(np.bool)& void_pixels
#    segmentation = segmentation.astype(np.bool)& void_pixels
    annotation = annotation.astype(np.bool) 
    segmentation = segmentation.astype(np.bool)
    
    whole_confuse = []
    for k in range(exp_config.num_bev_classes):
        temp_confuse = []
        cur_est = segmentation[...,k]
        for m in range(exp_config.num_bev_classes):
            temp_confuse.append(np.float32(np.sum((cur_est & annotation[...,m])& void_pixels)))
            
        whole_confuse.append(np.squeeze(np.array(temp_confuse)))
    
    return np.array(whole_confuse)


def iou_calculator(annotation, segmentation,vis_mask,occ_mask,mask_iou=False, void_pixels=None):
    """
    annotation : gt mask
    segmentation : method estimate
    """
#    if void_pixels is not None:
#        assert annotation.shape == void_pixels.shape, \
#            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
#        void_pixels = void_pixels.astype(np.bool)
#    else:
    
    
    if mask_iou:
        void_pixels = ~((occ_mask > 0.5) | (vis_mask < 0.5))
    else:
        void_pixels = vis_mask > 0.5
    
    
    annotation = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)
    
    inters = np.sum((segmentation & annotation) & void_pixels)
    union = np.sum((segmentation | annotation) & void_pixels)

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j, void_pixels
def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(np.bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(np.bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = np.squeeze(_seg2bmap(foreground_mask * np.logical_not(void_pixels)))
    gt_boundary = np.squeeze(_seg2bmap(gt_mask * np.logical_not(void_pixels)))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(np.bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def boundary_calculator(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)

    return f_res


def _pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap

def overlay_semantic_mask(im, ann, alpha=0.5, colors=None, contour_thickness=None):
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=np.int)
    if im.shape[:-1] != ann.shape:
        raise ValueError('First two dimensions of `im` and `ann` must match')
    if im.shape[-1] != 3:
        raise ValueError('im must have three channels at the 3 dimension')

    colors = colors or _pascal_color_map()
    colors = np.asarray(colors, dtype=np.uint8)

    mask = colors[ann]
    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]

    if contour_thickness:  # pragma: no cover
        import cv2
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours((ann == obj_id).astype(
                np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            cv2.drawContours(img, contours[0], -1, colors[obj_id].tolist(),
                             contour_thickness)
    return img


def train_saver(batch_image, batch_label, batch_dilated_label):
    
#    logging.error('TRAIN SAVER : ' + str(batch_image.shape))
    batch_image = np.squeeze(batch_image)
    batch_label = np.squeeze(batch_label)
    batch_dilated_label = np.squeeze(batch_dilated_label)
    n_images = batch_image.shape[0]
    
    if len(batch_image.shape) > 3:
    
        for k in range(n_images):
            query_image = batch_image[k,...]
            corrected_img = (query_image + 1)/2*255
            img_png=Image.fromarray(np.uint8(np.squeeze(corrected_img)))
            img_png.save(os.path.join(log_dir,'train_frame_'+str(k)+'_image'+'.png'))
            
            
            label = binary_to_png_convert(batch_label[k,...,0:exp_config.num_classes])
            img_png=Image.fromarray(np.squeeze(np.uint8(label)))
            img_png.save(os.path.join(log_dir,'train_frame_'+str(k)+'_label'+'.png'))
            
            
            
    else:
        query_image = np.squeeze(batch_image)
        corrected_img = (query_image + 1)/2*255
        img_png=Image.fromarray(np.uint8(np.squeeze(corrected_img)))
        img_png.save(os.path.join(log_dir,'train_frame_'+str(k)+'_image'+'.png'))
        
        
        label = binary_to_png_convert(batch_label[...,0:exp_config.num_classes])
        img_png=Image.fromarray(np.squeeze(np.uint8(label)))
        img_png.save(os.path.join(log_dir,'train_frame_'+str(k)+'_label'+'.png'))
            
        

#   temp=img[...,0]
#   img[...,0] = img[...,2]
#   img[...,2] = temp 
#   out_gt.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
   
            
def save_results(query_image,prev_whole_label,query_gt,seq,frame, val_path):

       
   corrected_img =  query_image 
   ann = np.squeeze(query_gt)
   img = overlay_semantic_mask(corrected_img, ann)
   img = img.astype(np.uint8)
   img_png=Image.fromarray(img)
   img_png = img_png.resize((800,450))
   img_png.save(os.path.join(val_path,'seq_'+str(seq)+'_frame_'+str(frame)+'_labels'+'.jpg'))
   
#   temp=img[...,0]
#   img[...,0] = img[...,2]
#   img[...,2] = temp 
#   out_gt.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
   
   ann = np.squeeze(prev_whole_label)
   img = overlay_semantic_mask(corrected_img, ann)
   img = img.astype(np.uint8)
   img_png=Image.fromarray(img)
   img_png = img_png.resize((800,450))
   img_png.save(os.path.join(val_path,'seq_'+str(seq)+'_frame_'+str(frame)+'_estimate'+'.jpg'))
   
  
#   out_estimate.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
def _clip_points_behind_camera(points, near_plane=1):
        """
        Perform clipping on polygons that are partially behind the camera.
        This method is necessary as the projection does not work for points behind the camera.
        Hence we compute the line between the point and the camera and follow that line until we hit the near plane of
        the camera. Then we use that point.
        :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
        :param near_plane: If we set the near_plane distance of the camera to 0 then some points will project to
            infinity. Therefore we need to clip these points at the near plane.
        :return: The clipped version of the polygon. This may have fewer points than the original polygon if some lines
            were entirely behind the polygon.
        """
        points_clipped = []
        indices_clipped = []
        # Loop through each line on the polygon.
        # For each line where exactly 1 endpoints is behind the camera, move the point along the line until
        # it hits the near plane of the camera (clipping).
        assert points.shape[0] == 3
        point_count = points.shape[1]
        for line_1 in range(point_count):
            line_2 = (line_1 + 1) % point_count
            point_1 = points[:, line_1]
            point_2 = points[:, line_2]
            z_1 = point_1[2]
            z_2 = point_2[2]

            if z_1 >= near_plane and z_2 >= near_plane:
                # Both points are in front.
                # Add both points unless the first is already added.
                if len(points_clipped) == 0 or all(points_clipped[-1] != point_1):
                    points_clipped.append(point_1)
                    indices_clipped.append(line_1)
                points_clipped.append(point_2)
                indices_clipped.append(line_2)
            elif z_1 < near_plane and z_2 < near_plane:
                # Both points are in behind.
                # Don't add anything.
                continue
            else:
                # One point is in front, one behind.
                # By convention pointA is behind the camera and pointB in front.
                if z_1 <= z_2:
                    point_a = points[:, line_1]
                    point_b = points[:, line_2]
                else:
                    point_a = points[:, line_2]
                    point_b = points[:, line_1]
                z_a = point_a[2]
                z_b = point_b[2]

                # Clip line along near plane.
                pointdiff = point_b - point_a
                alpha = (near_plane - z_b) / (z_a - z_b)
                clipped = point_a + (1 - alpha) * pointdiff
                assert np.abs(clipped[2] - near_plane) < 1e-6

                # Add the first point (if valid and not duplicate), the clipped point and the second point (if valid).
                if z_1 >= near_plane and (len(points_clipped) == 0 or all(points_clipped[-1] != point_1)):
                    points_clipped.append(point_1)
                points_clipped.append(clipped)
                if z_2 >= near_plane:
                    points_clipped.append(point_2)

        points_clipped = np.array(points_clipped).transpose()
        return points_clipped, indices_clipped

def view_points(points, view, normalize=True):
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]


    norm_const = points[2:3, :]
    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points,norm_const
def project_to_image(image1, cs1,cam_intrinsic):
    near_plane = 1
    
    # sample_point1 = np.array([int(3*image1.shape[1]/8),int(7*image1.shape[0]/8),1],np.float32)
    # sample_point2 = np.array([int(5*image1.shape[1]/8),int(7*image1.shape[0]/8),1],np.float32)
    # sample_point3 = np.array([int(3*image1.shape[1]/8),int(6*image1.shape[0]/8),1],np.float32)
    # sample_point4 = np.array([int(5*image1.shape[1]/8),int(6*image1.shape[0]/8),1],np.float32)
    
    # base_sample_points = np.stack([sample_point1,sample_point2,sample_point3,sample_point4],axis=-1)
    
    # source_points = np.float32(base_sample_points[0:2,:].T)
    # image1=np.zeros((196,200),np.float32)
    sample_point1 = np.array([int(3*image1.shape[1]/8),int(2*image1.shape[0]/8),0],np.float32)
    sample_point2 = np.array([int(5*image1.shape[1]/8),int(2*image1.shape[0]/8),0],np.float32)
    sample_point3 = np.array([int(3*image1.shape[1]/8),int(6*image1.shape[0]/8),0],np.float32)
    sample_point4 = np.array([int(5*image1.shape[1]/8),int(6*image1.shape[0]/8),0],np.float32)
    
    sample_points = np.stack([sample_point1,sample_point2,sample_point3,sample_point4],axis=-1)
    
    source_points = np.float32(np.copy(sample_points[0:2,:]).T)
    
    sample_points[1,:] = image1.shape[0] - sample_points[1,:]
    
    sample_points[0,:] = -(sample_points[0,:] - image1.shape[1]/2)*exp_config.resolution
    sample_points[1,:] = sample_points[1,:]*exp_config.resolution
    sample_points=np.stack([sample_points[1,:],sample_points[0,:],sample_points[2,:]])

    # source_points[1,:] = image1.shape[0] - source_points[1,:]
    
    # inverse_intrinsic =  np.linalg.inv(cam_intrinsic)
    
    # pr = Quaternion(pose1['rotation']).rotation_matrix.T
    
    # inverse_pr = np.linalg.inv(pr)
    
    # pt = np.array(pose1['translation']).reshape((-1, 1))
    
    cr = Quaternion(cs1['rotation']).rotation_matrix.T
    # cr = Quaternion([0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755]).rotation_matrix.T
    # inverse_cr = np.linalg.inv(cr)
    
    # ct = np.array([1.70079118954, 0.0159456324149, 1.51095763913]).reshape((-1, 1))
    ct = np.array(cs1['translation']).reshape((-1, 1))
    

    # sample_points = base_sample_points.T
   
    # Transform into the camera.
    points = sample_points - ct
    points = np.dot(cr, points)
    
    
    # points,_ = _clip_points_behind_camera(points, near_plane)
    
    camera_points, norm_const = view_points(points, cam_intrinsic, normalize=True)


    
    estimated_transform = cv2.getPerspectiveTransform(source_points,np.float32(camera_points.T[:,:2]))
    
    
    # total_image_size=exp_config.original_image_size
    # warped_image_ref = cv2.warpPerspective(image1,estimated_transform,total_image_size,flags=cv2.INTER_LINEAR)
   
#        logging.error('PROJECT TO GROUND REF WARPED')
    
    return estimated_transform       
def main():

    continue_run = True
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
        continue_run = False

    # Copy experiment config file
    shutil.copy(exp_config.__file__, log_dir)

    run_training(continue_run)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(
    #     description="Train a neural network.")
    # parser.add_argument("CONFIG_PATH", type=str, help="Path to config file (assuming you are in the working directory)")
    # args = parser.parse_args()

    main()
