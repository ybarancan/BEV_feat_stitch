
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

from background_generator import BackgroundGenerator

import glob

import random

from PIL import Image

from deeplab import common

import mem_net

from multiprocessing.dummy import Pool as ThreadPool 

import utils


from nuscenes.nuscenes import NuScenes
#from nuscenes.nuscenes import NuScenesExplorer
from nuscenes.map_expansion.map_api import NuScenesMap

import token_splits

from experiments import nuscenes_objects_base as exp_config

########################################################################################
means_image = np.array([123.68, 116.779, 103.939], dtype=np.single)

nusc = NuScenes(version='v1.0-trainval', dataroot=exp_config.nuscenes_root, verbose=True)
scenes = nusc.scene

nusc_map_sin_onenorth = NuScenesMap(dataroot= exp_config.nuscenes_root, map_name='singapore-onenorth')
nusc_map_sin_hollandvillage = NuScenesMap(dataroot=exp_config.nuscenes_root, map_name='singapore-hollandvillage')
nusc_map_sin_queenstown = NuScenesMap(dataroot=exp_config.nuscenes_root, map_name='singapore-queenstown')
nusc_map_bos = NuScenesMap(dataroot=exp_config.nuscenes_root, map_name='boston-seaport')

global_const = 3.99303084

total_label_slices = exp_config.num_classes + 2

target_dir = exp_config.nuscenes_processed_root

exp_config.batch_size=1

do_eval_on_whole_videos = True

use_deeplab = True
starting_from_cityscapes = True
starting_from_imagenet = False

do_eval_frames=[3,4,5]

use_balanced_loss=True

use_binary_loss = True

num_frames=exp_config.num_frames

reference_frame_index = 1

n_frames_per_seq = exp_config.num_frames

n_seqs = n_frames_per_seq-num_frames+1
softmax_aggregation_testing = True

use_occlusion=exp_config.use_occlusion

BATCH_SIZE = exp_config.batch_size
"""
If inception pre-process is used, the inputs to query encoders are corrected through vgg processing in the tensorflow part.
Query encoders do not use masks, thus they can be simply propagated through the Resnet. Memory encoders need to be handled 
differently since if the image's range is 0-255 and mask is 0-1 then the mask is not effective through simple addition before
batch norm. 
If root block is included, inception_preprocess should be set to False.
"""

use_inception_preprocess = True
freeze_batch_norm_layers = True
multiply_labels=True
include_root_block=True
apply_same_transform='new'
rec_count = exp_config.max_tries
apply_intermediate_loss = False

log_dir = os.path.join('/scratch_net/catweazle/cany/mapmaker_github/logdir/deeplab'+str(use_deeplab))

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
    warp_trans1 = utils.tensorflow_project_to_ground(exp_config, image,np.zeros((int(exp_config.camera_image_patch_size[0]/(4*exp_config.downsample_ratio)),int(exp_config.camera_image_patch_size[1]/(4*exp_config.downsample_ratio)))),poserecord_ref, cs_record_ref,poserecord_cur,cs_record_cur, cam_intrinsic,reference_frame=is_reference_sample)
    warp_trans2 = utils.tensorflow_project_to_ground(exp_config, image,np.zeros((int(exp_config.camera_image_patch_size[0]/(8*exp_config.downsample_ratio)),int(exp_config.camera_image_patch_size[1]/(8*exp_config.downsample_ratio)))),poserecord_ref, cs_record_ref,poserecord_cur,cs_record_cur, cam_intrinsic,reference_frame=is_reference_sample)
    warp_trans3 = utils.tensorflow_project_to_ground(exp_config, image,np.zeros((int(exp_config.camera_image_patch_size[0]/(16*exp_config.downsample_ratio)),int(exp_config.camera_image_patch_size[1]/(16*exp_config.downsample_ratio)))),poserecord_ref, cs_record_ref,poserecord_cur,cs_record_cur, cam_intrinsic,reference_frame=is_reference_sample)
        
    
    
    warped_img, warped_cover, warped_label, coordinate_transform = utils.project_to_ground(exp_config, image,label,poserecord_ref, cs_record_ref,poserecord_cur,cs_record_cur, cam_intrinsic,vis_mask,reference_frame=is_reference_sample)
    
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
        pre_warped_img = utils.inception_preprocess(warped_img)
        pre_img=cropped_img - means_image 
#        pre_warped_img=warped_img - means_image 

    else:
        pre_img=cropped_img  
        pre_warped_img= utils.inception_preprocess(warped_img)

#    logging.error('Pre img shape ' + str(pre_img.shape))
  
    return (pre_img, np.float32(temp_label),pre_warped_img, np.float32(bev_label),warped_cover,coordinate_transform,np.reshape(warp_trans1,[-1])[0:8],np.reshape(warp_trans2,[-1])[0:8],np.reshape(warp_trans3,[-1])[0:8])


def run_training(continue_run):

#    train_file ='C:\\winpython\\WPy-3670\\codes\\davis2017\\DAVIS\\ImageSets\\2017\\train.txt'
#    data_images_path ='C:\\winpython\\WPy-3670\\codes\\davis2017\\DAVIS\\JPEGImages\\480p\\drone'
    
    
    logging.error('EXPERIMENT : ' + str(exp_config.experiment_name))
    logging.error('THIS IS : ' + str(log_dir))
    
    val_tokens = token_splits.VAL_SCENES
    train_tokens = token_splits.TRAIN_SCENES
    
    
    batch_indices = np.arange(len(train_tokens))
    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name)


    init_step = 0
    # Load data
    

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
        
        
        ref_bev_labels_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,exp_config.label_patch_size[1],exp_config.label_patch_size[0],exp_config.num_bev_classes + 2], name='ref_bev_labels')          
        
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
        
        alpha_pos = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(exp_config.image_object_positive_weights,axis=0),axis=0),axis=0), tf.float32)


        side_obj_loss, _ = mem_net.full_object_loss(side_obj_softmaxed,image_objects_label_placeholder,tf.slice(image_labels_placeholder,[reference_frame_index,0,0,exp_config.num_classes+1],[1,-1,-1,-1]),exp_config,alpha_pos,weight=True,weight_vector=None, focal=True)
        
        side_seg_loss0, side_alpha0 = mem_net.contrastive_sigmoid_loss(side_mask_logits,image_labels_placeholder,exp_config,weight=True)
        
  
        
        side_occ_loss0 = mem_net.occlusion_loss(side_occ_est_logits, tf.squeeze(tf.slice(image_labels_placeholder,[0,0,0,exp_config.num_classes+1],[-1,-1,-1,-1]),axis=-1))
        
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
        
        bigger_resized_combined_projected_estimates = tf.image.resize(
                combined_projected_estimates, [int(exp_config.patch_size[1]/4),int(exp_config.patch_size[0]/4)] ,method='bilinear',name='bigger_projected_estimates_resize'  )
 
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
        
#        bev_total_relative_endpoints = [combined_end]
        bev_total_relative_endpoints = [tf.concat([combined_end,bigger_resized_combined_projected_estimates],axis=-1)]
        
       
        
        total_input = tf.concat([ resized_combined_projected_estimates,bev_total_backbone_out],axis=-1)
        
        
        
        
        static_logits, static_masks,object_logits, object_masks = mem_net.my_bev_object_decoder(bev_total_relative_endpoints,total_input,exp_config,apply_softmax=True,reuse=False)
        
        # object_logits, object_masks = mem_net.my_bev_static_decoder(bev_total_relative_endpoints,total_input,exp_config,reuse=False)
        
        
        cur_covers = tf.slice(resized_covers,[0,0,0,0],[1,-1,-1,-1])
        
        # seg_loss0, alpha0 = mem_net.bev_object_loss(mask_logits,tf.slice(ref_bev_labels_placeholder,[0,0,0,0],[1,-1,-1,-1]),cur_covers,exp_config,weight=True)
        alpha_pos = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(exp_config.bev_positive_weights,axis=0),axis=0),axis=0), tf.float32)
        alpha_neg = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(exp_config.bev_negative_weights,axis=0),axis=0),axis=0), tf.float32)

        masks = tf.concat([static_masks,object_masks],axis=-1)
        seg_loss0, alpha0 = mem_net.full_modified_bev_object_loss(masks,ref_bev_labels_placeholder,cur_covers,exp_config,alpha_pos,alpha_neg,weight=True)
        

        '''
        LOSSES ADDED
        '''
        mean_side_seg_loss0 = tf.reduce_mean(side_seg_loss0)
        mean_side_obj_loss = tf.reduce_mean(side_obj_loss)
        mean_seg_loss0 = tf.reduce_mean(seg_loss0)
#            mean_seg_loss1 = tf.reduce_mean(seg_loss1)
        
        occ_loss = tf.constant(0)
        
        recon_loss = mean_seg_loss0 
        
        side_loss = mean_side_seg_loss0 + 0.001*tf.reduce_mean(side_occ_loss0) + 10*mean_side_obj_loss
        l2_loss_vars = []
        # l2_loss_vars = []
        trainable_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in trainable_vars:
            cur_name = var.op.name.split('/')[-1]
            # logging.error('CUR NAME ' + cur_name)
            if not (('bias' in cur_name) | ('_b' in cur_name) | ('gamma' in var.op.name) | ('beta' in var.op.name)):
                l2_loss_vars.append(var)
            
        # logging.error(str(l2_loss_vars))
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in l2_loss_vars])/len(l2_loss_vars)
        loss = recon_loss + 0.0001*lossL2 + 2*side_loss 
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder)
#            
#           
        
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        trainable_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        optimizer_variables=[]

        backbone_optimizer_variables=[]
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
        for var in all_vars:
            if (not ('my_bev_object_decoder' in var.op.name)) :
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
#                   
            else:
                logging.error('NON BACKBONE VAR ' + str(var))
                optimizer_variables.append(var)
        
        logging.error('NON DECODER VARS '+ str(non_decoder_vars))
        
        logging.error('NUMBER OF ALL PARAMETERS: ' + str(np.sum([np.prod(v.get_shape().as_list()) for v in optimizer_variables])))
        logging.error('NUMBER OF BACKBONE PARAMETERS: ' + str(np.sum([np.prod(v.get_shape().as_list()) for v in backbone_optimizer_variables])))
            
        # to_load_saver = tf.train.Saver(var_list=non_decoder_vars,max_to_keep=2)
        
        gvs = optimizer.compute_gradients(loss,var_list=optimizer_variables)
        
        capped_gvs = get_clipped_grads(gvs)
        
        network_train_op_total = optimizer.apply_gradients(capped_gvs)
                
        
        # to_load_saver = tf.train.Saver(var_list=to_load_vars,max_to_keep=2)
        saver = tf.train.Saver(max_to_keep=2)
        
        saver_best_loss = tf.train.Saver(max_to_keep=2)
        init = tf.global_variables_initializer()
        sess.run(init)
        
    
        if use_deeplab & starting_from_cityscapes:
            load_path = '/scratch_net/catweazle/cany/cityscapes_deeplab/model.ckpt'
      
            imagenet_query_saver.restore(sess, load_path)
        
        elif starting_from_imagenet:
            load_path1 = os.path.join('/scratch_net/catweazle/cany/resnet50/resnet_v1_50_2016_08_28/resnet_v1_50_1.ckpt')
           
        
            imagenet_image_saver.restore(sess, load_path1)

        else:
            load_path = os.path.join(log_dir ,'checkpoints','routine-99999')

            saver.restore(sess,load_path)
          
        sess.run(mem_net.interp_surgery(tf.global_variables()))
        
        
        
        init_step = 0
        
        
        time2=0
        time3 = 0
        start_epoch = 0
        step = init_step
        curr_lr = exp_config.learning_rate

        
        curr_lr = 1e-05
        

        max_epoch = 3000
        best_mean = 0.2
        
#            i1_value_list = []
#            i2_value_list = []
        loss_value_list=[]
        occ_loss_value_list=[]
        recon_loss_value_list=[]
        reg_loss_value_list=[]
        
        side_loss_value_list=[]
        side_occ_loss_value_list=[]
        side_recon_loss_value_list=[]
        side_obj_loss_value_list=[]
        seg_v_list0=[]
      
        boundary_loss_value_list=[]
        
        for epoch in range(start_epoch,max_epoch):
   
            
            if epoch % 40 == 0:
                curr_lr = 0.9*curr_lr
            
            logging.error('EPOCH : ' + str(epoch))
            # Update learning rate if necessary


            # Iterate over batches
            random.shuffle(train_tokens)
            
            batch_indices_list = []
                
            for k in range(BATCH_SIZE):

#                    
#                    random.shuffle(batch_indices)
                batch_indices_list.append(batch_indices[k::BATCH_SIZE])
#                
#                    logging.error(str(batch_indices_list[-1]))
#                
            generators_list=[]
  
            max_interval_between_frames = 3
            for k in range(BATCH_SIZE):
                generators_list.append( BackgroundGenerator(iterate_minibatches(train_tokens,max_interval_between_frames, reference_frame_index=reference_frame_index, n_frames_per_seq= n_frames_per_seq,
                                                                     batch_size=BATCH_SIZE)))

                try:   
                    end_of_epoch=False
                    while (not end_of_epoch):
                            end_of_epoch=False
    #                     
                            for k in range(BATCH_SIZE):
                                temp_next = generators_list[k].next()
                                if temp_next == None:
                                    end_of_epoch=True
                                    break
        #            
                                else:
                                    
                                    temp1, temp2, temp3,temp4,temp5, temp6,temp7,temp8, temp9,temp10,temp11,batch_channel_bev_images,batch_ref_bev_labels,batch_image_objects = temp_next
                                    batch_image=temp1
                                    batch_label=temp2                        
      
                             
                                    batch_bev_covers=temp5    
                                    batch_transforms = temp6
                                    batch_tf_transforms1=temp7
                                    batch_tf_transforms2=temp8
                                    batch_tf_transforms3=temp9
                                    batch_separate_covers = temp10
                                    batch_coordinate_transforms = temp11
                            if end_of_epoch:
    #                        
                                break
                            
    #                        batch_bev_covers = batch_bev_covers[...,0]
                            
                            if step % 5000 == 4999:
                                saver.save(sess,
                                 os.path.join(log_dir,
                                              'checkpoints',
                                              'routine'),
                                 global_step=step)
    
                            feed_dict = {
                                    
                            learning_rate_placeholder:curr_lr,
                            training_time_placeholder: True,
                            my_training_placeholder:True,
                           
                            images_placeholder:batch_image,
                            image_objects_label_placeholder:batch_image_objects,
                            image_labels_placeholder:batch_label,
                            covers_placeholder:batch_bev_covers,
                            bev_transforms_placeholder:batch_transforms,
                            separate_covers_placeholder : batch_separate_covers,
                            ground_transforms_placeholder1:batch_tf_transforms1,
                            ground_transforms_placeholder2:batch_tf_transforms2,
                            ground_transforms_placeholder3:batch_tf_transforms3,
                            coordinate_ground_transforms_placeholder:batch_coordinate_transforms,
                            
                            channel_bev_images_placeholder:batch_channel_bev_images,
                            ref_bev_labels_placeholder: batch_ref_bev_labels
                            
                            
                            }
                            time1 = time.time()
                            
                            data_loading_time = time1-time3
                            
                            _ = sess.run(network_train_op_total, feed_dict=feed_dict)
    #                       
                            time2 = time.time()
                            
                            if step % 100 == 0:
                                
        
        
                                loss_value,side_obj_v,recon_v,occ_v,reg_v,seg_loss_v0, mean_side_seg_loss0_v,side_occ_loss0_v,side_loss_v=\
                                sess.run([loss,side_obj_loss,recon_loss,occ_loss,lossL2,mean_seg_loss0, mean_side_seg_loss0, side_occ_loss0, side_loss], feed_dict=feed_dict)
    #                           
                                recon_loss_value_list.append(recon_v)
                                reg_loss_value_list.append(reg_v)
                                occ_loss_value_list.append(occ_v)
                                
                                side_loss_value_list.append(side_loss_v)
                                side_occ_loss_value_list.append(side_occ_loss0_v)
                                side_recon_loss_value_list.append(mean_side_seg_loss0_v)
                                side_obj_loss_value_list.append(side_obj_v)
    #                            side_loss_value_list.append(side_recon_v)
                                loss_value_list.append(loss_value)
                                
                                seg_v_list0.append(seg_loss_v0)         
    #                         
                              
                            # Write the summaries and print an overview fairly often.
                            if step % 1000 == 0:
     
                                logging.error('Step %d: loss= %.4f, rec= %.4f, reg= %.4f, occ= %.4f ' % (step, np.mean(loss_value_list),np.mean(recon_loss_value_list),np.mean(reg_loss_value_list),np.mean(occ_loss_value_list)))
                                
                                logging.error('Step %d: side loss = %.4f, side rec = %.4f, side occ = %.4f, side obj = %.4f, bound = %.4f  ' % (step, np.mean(side_loss_value_list),np.mean(side_recon_loss_value_list),np.mean(side_occ_loss_value_list),np.mean(side_obj_loss_value_list), np.mean(boundary_loss_value_list)))
                                
                                
                                logging.error('Time it took for optimization : ' + str(time2-time1) + ' and data loading: '+ str(data_loading_time) )
                                
    #  
                                loss_value_list=[]
                                occ_loss_value_list=[]
                                recon_loss_value_list=[]
                                reg_loss_value_list=[]
                                
                           
                                boundary_loss_value_list=[]
                                seg_v_list0=[]
                              
                                side_loss_value_list=[]
                                side_occ_loss_value_list=[]
                                side_obj_loss_value_list=[]
                                side_recon_loss_value_list=[]
                                
                              
                                
                            time3 = time.time()
                            if step % exp_config.val_eval_frequency == (exp_config.val_eval_frequency - 1):
    
     #              
     #
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
                                 
                                 combined_projected_estimates,
                                step,training_time_placeholder,val_folder_path=validation_res_path)
                                 overall_mean = np.mean(np.array(val_res))
                                 logging.error('Overall mean : ' + str(overall_mean))
    
                                 if overall_mean > best_mean:
                                     best_mean = np.copy(overall_mean)
                                     logging.error('New best')
                                     saver_best_loss.save(sess,
                                                         os.path.join(log_dir,
                                               'checkpoints',
                                               'best-'+str(np.uint8(np.floor(overall_mean*100)))),
                                                 global_step=step)
    # ####                            
    #                                
                                     
                                 
                            step = step + 1

                except Exception as e:
                  
                    logging.error(str(e))
                    
                    continue
   
    


def eval_iterator(my_scene,cur_index, reference_frame_index, single_frame=False): 
    n_seqs = 1
    current_dir = os.path.join(target_dir,'scene'+my_scene)
    
    pool = ThreadPool(n_seqs*num_frames) 

    all_images_list = sorted(glob.glob(os.path.join(current_dir,'img*.png')))
    all_labels_list = sorted(glob.glob(os.path.join(current_dir,'label*.png')))
    
    first_frame = cur_index
    
    frame_ids=[]
    frame_ids.append(first_frame)
    
    if single_frame:
    
        for frame_number in range(1,n_frames_per_seq):
            frame_ids.append(first_frame )
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
        if k < (n_seqs - 1):
            cur_sample = nusc.sample[all_sample_inds[frame_ids[reference_frame_index + k]]]
            next_sample = nusc.sample[all_sample_inds[frame_ids[reference_frame_index + k+1]]]
            
            my_trans = np.copy(utils.tensorflow_project_bev_to_bev(nusc, exp_config, cur_sample,next_sample))
            my_trans = np.reshape(my_trans,[-1])[0:8]
            transforms_list.append(my_trans)
            
        cur_ref_sample = nusc.sample[all_sample_inds[frame_ids[reference_frame_index + k]]]
        reference_samples.append(cur_ref_sample)
        
        cam_token_cur = cur_ref_sample['data']['CAM_FRONT']
        cam_record_cur = nusc.get('sample_data', cam_token_cur)
        
        bev_label = np.array(Image.open( os.path.join( exp_config.nuscenes_bev_root,  
                                   cam_record_cur['token'] + '.png')),np.int32)
        
        bev_label = np.flipud(bev_label)
        
        bev_label = decode_binary_labels(bev_label, exp_config.num_bev_classes+1)
        bev_labels_list.append(bev_label)
        
    
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
   
    

    return seq_images_ar, seq_labels_ar,bev_images_ar,bev_labels_ar,bev_covers_ar,np.zeros((1,8)), bev_transforms_ar1,bev_transforms_ar2,bev_transforms_ar3,coordinate_transforms_ar,np.stack(bev_labels_list,axis=0),True


def overall_eval_iterator(my_scene,cur_index, reference_frame_index, single_frame=False):
    
    # logging.error('SINGLE FRAME ' + str(single_frame))
    seq_images_ar, seq_labels_ar, bev_images_ar,bev_labels_ar,bev_covers_ar, transforms_ar,tf_transforms1,tf_transforms2,tf_transforms3, coordinate_transforms_ar,real_ref_bev_labels,went_well =  eval_iterator(my_scene,cur_index,reference_frame_index,single_frame=single_frame)
        

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
   
    return seq_images_ar, seq_labels_ar, fin_bev_images, fin_bev_labels,fin_covers , transforms_ar,tf_transforms1,tf_transforms2,tf_transforms3, bev_covers_ar, coordinate_transforms_ar,to_return_bev_images,real_ref_bev_labels




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
                       
                        combined_projected_estimates,
                        iteration,training_time_placeholder,val_folder_path=validation_res_path):

   
    logging.error('Started evaluation')
    
    
    res_strings=[]
    all_j1s=[]

    for my_scene_token in val_tokens[22:30]:
        
        scene_results=[]
      
        current_dir = os.path.join(target_dir,'scene'+my_scene_token)
        
        images = sorted(glob.glob(os.path.join(current_dir,'img*.png')))
        
        name_of_seq = my_scene_token
        
        
        
        if not os.path.exists(os.path.join(validation_res_path,name_of_seq)):
            os.makedirs(os.path.join(validation_res_path,name_of_seq))
        
        
        
        for frame_number in range(len(images)):
#            logging.error('FRAME NUMBER ' + str(frame_number))
            
            if frame_number < reference_frame_index:
                
            
                batch_image, batch_label, batch_bev_images, batch_bev_labels,batch_bev_covers ,batch_transforms,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels =\
                overall_eval_iterator(my_scene_token,frame_number,reference_frame_index,single_frame=True)
            elif (frame_number >= (len(images) - (num_frames - reference_frame_index - 1))):
            
                batch_image, batch_label, batch_bev_images, batch_bev_labels,batch_bev_covers ,batch_transforms,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels =\
                overall_eval_iterator(my_scene_token,frame_number,reference_frame_index,single_frame=True)
                
            else:
                
                batch_image, batch_label, batch_bev_images, batch_bev_labels,batch_bev_covers ,batch_transforms,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels =\
                overall_eval_iterator(my_scene_token,int(frame_number - reference_frame_index),reference_frame_index,single_frame=False)
                
            
            
        
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

            masks_v,side_masks_v,side_occ_masks_v, projected_estimates_v,combined_projected_estimates_v,side_obj_softmaxed_v = sess.run([masks,
                        side_masks,side_occ_masks,projected_estimates,
                        combined_projected_estimates,side_obj_softmaxed], feed_dict=feed_dict)
#            logging.error('DEC OUTPUT ' + str(temp_dec_output.shape))
            
                
            sample_results=[]
            squeezed_masks = np.squeeze(masks_v)
            static_estimates = np.uint8(squeezed_masks[...,:exp_config.num_static_classes] > 0.5)
            
            temp_object_estimates = squeezed_masks[...,exp_config.num_static_classes:]
            
            # logging.error('TEMP OBJ ' + str(temp_object_estimates))
            temp_object_estimates = np.argmax(temp_object_estimates,axis=-1)        
            object_estimates = np.zeros((temp_object_estimates.shape[0],temp_object_estimates.shape[1],exp_config.num_object_classes))
            for k in range(exp_config.num_object_classes):
                object_estimates[...,k] = np.uint8(temp_object_estimates == k)
            
            hard_estimates = np.concatenate([static_estimates,object_estimates],axis=-1)
            for k in range(exp_config.num_bev_classes):
                
                # bev_estimate = np.squeeze(masks_v[...,k])
                # hard_estimate_list.append(np.uint8(bev_estimate > 0.5))
                
                all_stats , void_pixels= utils.get_all_stats(np.squeeze(batch_ref_bev_labels[...,k]), hard_estimates[...,k],np.squeeze(batch_ref_bev_labels[...,exp_config.num_bev_classes]),mask_iou=exp_config.use_occlusion)
                sample_results.append(all_stats)
                
            scene_results.append(np.array(sample_results))
            # occ_scene_results.append(occ_all_stats)
            
            
        
        seq_j1 = np.array(scene_results)
        
        # logging.error('SEQ J SHAPE ' + str(seq_j1.shape))
        all_j1s.append(np.squeeze(seq_j1))
   
        temp_string = "Iteration : " + str(iteration) + " : Scene " + str(my_scene_token)+ " - j1: " + str(np.mean(seq_j1,axis=0)[...,0]) 
        
        
        res_strings.append(temp_string)
        logging.error(temp_string)

    
    tot_j1 = np.concatenate(all_j1s,axis=0)
  
    j = tot_j1[...,0]
    
    tp = tot_j1[...,2]
    fp = tot_j1[...,3]
    fn = tot_j1[...,4]
    tn = tot_j1[...,5]
    
    
    
    tp_rate = np.sum(tp,axis=0)/(np.sum(tp,axis=0) +  np.sum(fn,axis=0) + 0.0001)
    fp_rate = np.sum(fp,axis=0)/( np.sum(fp,axis=0) + np.sum(tn,axis=0) + 0.0001)
    
    
    tp_rate = np.sum(tp,axis=0)/(np.sum(tp,axis=0) +  np.sum(fn,axis=0) + 0.0001)
    fp_rate = np.sum(fp,axis=0)/( np.sum(fp,axis=0) + np.sum(tn,axis=0) + 0.0001)
    precision = np.sum(tp,axis=0)/(np.sum(tp,axis=0) +  np.sum(fp,axis=0) + 0.0001)

    
    take_all_j = np.mean(j,axis=0)
    confuse_iou = np.sum(tp,axis=0)/(np.sum(tp,axis=0) + np.sum(fp,axis=0) + np.sum(fn,axis=0) + 0.0001)
    
    temp_string = 'Bev framewise j : ' + str(take_all_j)+ ' , Bev confuse j : ' + str(confuse_iou) + '\n' +\
    ' Bev tp_rate : ' + str(tp_rate)+ ' Bev fp_rate : ' + str(fp_rate)+ ' Bev precision : ' + str(precision)+ '\n'
    
    
    logging.error(temp_string)
    res_strings.append(temp_string)
    utils.write_to_txt_file(os.path.join(log_dir,'val_results.txt'),res_strings)
    return confuse_iou


def standard_iterate_minibatches(my_scene,max_interval_between_frames,
                                                                          reference_frame_index,
                                                                         n_frames_per_seq=3,
                                                                         batch_size=1): 

    n_seqs = n_frames_per_seq-num_frames+1
    try:
        current_dir = os.path.join(target_dir,'scene'+my_scene)
        
#        logging.error('Cur directory : ' + str(current_dir))
        pool = ThreadPool(n_seqs*num_frames) 

        
        all_images_list = sorted(glob.glob(os.path.join(current_dir,'img*.png')))
        all_labels_list = sorted(glob.glob(os.path.join(current_dir,'label*.png')))
        
#
        
        n_frames_in_scene = len(all_images_list)
        seq_length = n_frames_in_scene
        frame_ids=[]
        first_frame = random.randint(0,n_frames_in_scene-n_frames_per_seq)
        frame_ids.append(first_frame)
       
        for frame_number in range(1,n_frames_per_seq):
            frame_ids.append(random.randint(frame_ids[-1]+1, np.min([seq_length-(n_frames_per_seq-frame_number),frame_ids[-1]+max_interval_between_frames])))

            
        pairs = []
        
        
        scene_token = current_dir.split('/')[-1][5:]
        
        my_scene = nusc.get('scene', scene_token)
        
        first_sample_token = my_scene['first_sample_token']
        last_sample_token = my_scene['last_sample_token']
        first_sample_ind = nusc.getind('sample',first_sample_token)
        last_sample_ind = nusc.getind('sample',last_sample_token)
        
        all_sample_inds = np.arange(first_sample_ind,last_sample_ind+1)
            
        reference_samples = []
        
#        logging.error('STARTING BEV TO BEV')
        
        bev_labels_list=[]
        
        for k in range(n_seqs):
      
                
            cur_ref_sample = nusc.sample[all_sample_inds[frame_ids[k+reference_frame_index]]]
            reference_samples.append(cur_ref_sample)
            
            cam_token_cur = cur_ref_sample['data']['CAM_FRONT']
            cam_record_cur = nusc.get('sample_data', cam_token_cur)
            
            bev_label = np.array(Image.open( os.path.join(exp_config.nuscenes_bev_root,  
                                       cam_record_cur['token'] + '.png')),np.int32)
            
            bev_label = np.flipud(bev_label)
            
            cs_record_cur = nusc.get('calibrated_sensor', cam_record_cur['calibrated_sensor_token'])
            cam_intrinsic = np.array(cs_record_cur['camera_intrinsic'])
            
            bev_label = decode_binary_labels(bev_label,exp_config.num_bev_classes+1)
            vis_mask = np.flipud(utils.get_visible_mask(cam_intrinsic, cam_record_cur['width'], 
                                   exp_config.map_extents, exp_config.map_resolution))
            
            # logging.error('VIS MASK ' + str(vis_mask.shape))
            # logging.error('BEV LABEL MASK ' + str(bev_label.shape))
            bev_label = np.concatenate([bev_label,np.expand_dims(vis_mask,axis=-1)],axis=-1)
            bev_labels_list.append(bev_label)
            
            
            # np.savez('/home/cany/image_trans_stuff.npy',cs_record_cur,cam_intrinsic)
            
            to_image_transform = utils.project_to_image(exp_config, np.zeros((exp_config.project_base_patch_size[1],exp_config.project_base_patch_size[0])),cs_record_cur,cam_intrinsic)

            
            
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
        
        image_objects = cv2.warpPerspective(np.squeeze(real_ref_bev_labels[...,4:-2]),to_image_transform,exp_config.original_image_size,flags=cv2.INTER_LINEAR)
        image_objects= cv2.resize(image_objects,(int(exp_config.camera_image_patch_size[1]/4),int(exp_config.camera_image_patch_size[0]/4)), interpolation = cv2.INTER_LINEAR)
        image_objects = np.expand_dims(np.float32(image_objects > 0.5),axis=0)
        
        image_objects = np.concatenate([image_objects,np.clip(1-np.sum(image_objects,axis=-1,keepdims=True),0,1)],axis=-1)
        
        yield seq_images_ar, seq_labels_ar, fin_bev_images, fin_bev_labels,fin_covers , transforms_ar,tf_transforms1,tf_transforms2,tf_transforms3, bev_covers_ar, coordinate_transforms_ar,to_return_bev_images,real_ref_bev_labels,image_objects

def main():

    continue_run = True
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
        continue_run = False

    # Copy experiment config file
    shutil.copy(exp_config.__file__, log_dir)

    run_training(continue_run)


if __name__ == '__main__':



    main()
