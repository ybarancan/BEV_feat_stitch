
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

import random

from PIL import Image

from deeplab import common

import mem_net

import utils
from multiprocessing.dummy import Pool as ThreadPool 

from argoverse.data_loading.argoverse_tracking_loader \
    import ArgoverseTrackingLoader
import dataset.argoverse_token_splits as token_splits

from experiments import argoverse_objects_exp as exp_config

########################################################################################
means_image = np.array([123.68, 116.779, 103.939], dtype=np.single)

total_label_slices = exp_config.num_classes + 2

train_path = os.path.expandvars(exp_config.argo_track_path)
train_loader = ArgoverseTrackingLoader(train_path)

target_dir = exp_config.argo_labels_path

camera = "ring_front_center"

all_frames_list=[]

total_n = 0

for my_scene_id in range(len(token_splits.TRAIN_LOGS)):
    
    scene = train_loader.get(token_splits.TRAIN_LOGS[my_scene_id])


    n_frames_in_scene = scene.num_lidar_frame
    
    all_frames_list.append(np.copy(n_frames_in_scene))
    
    total_n = total_n + n_frames_in_scene
    
train_frame_list = np.arange(total_n)
    

all_frames_cumulative = np.cumsum(np.array(all_frames_list))

logging.error('ALL FRAMES LIST ')
logging.error(str(all_frames_list))

logging.error('ALL FRAMES CUMULATIVE ')
logging.error(str(all_frames_cumulative))

exp_config.batch_size=1


use_deeplab = True
starting_from_cityscapes =True
starting_from_imagenet =False


num_frames=exp_config.num_frames

reference_frame_index = 1

n_frames_per_seq = exp_config.num_frames

n_seqs = n_frames_per_seq-num_frames+1

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


log_dir = exp_config.log_dir
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
     
       
    camera_channel = 'ring_front_center'
    image_path,label_path, vis_mask,calib_cur,pose_ref, pose_cur , is_reference_sample= pair
    
    calib_ref = calib_cur
    cam_intrinsic = calib_cur.K[:,:3]
    
    img = Image.open(image_path)
    img.load()
    
    encoded_labels = np.flipud(np.array(Image.open(label_path),np.int32))
        
    num_class = exp_config.num_bev_classes
    bev_labels = decode_binary_labels(encoded_labels, num_class+ 1)
    
    bev_labels = np.concatenate([np.copy(bev_labels[...,:exp_config.num_bev_classes]),np.copy(vis_mask),vis_mask*(1-np.copy(np.expand_dims(bev_labels[...,exp_config.num_bev_classes],axis=-1)))],axis=-1)
    
    to_image_transform = utils.argoverse_project_to_image(exp_config, np.zeros_like(bev_labels),calib_ref)
    
    image_labels = cv2.warpPerspective(np.float32(bev_labels),to_image_transform,exp_config.original_image_size,flags=cv2.INTER_LINEAR)
    image_labels = np.uint8(image_labels > 0.3)
    
    # image_objects= cv2.resize(image_objects,(int(exp_config.camera_image_patch_size[1]/4),int(exp_config.camera_image_patch_size[0]/4)), interpolation = cv2.INTER_LINEAR)
    
    
    # save_array(np.expand_dims(image_labels,axis=0),'temp_res',is_rgb=False)
    image=np.array(img, dtype=np.uint8)

    
    warp_trans1 = utils.argoverse_tensorflow_project_to_ground(exp_config, image,np.zeros((int(exp_config.camera_image_patch_size[0]/(4*exp_config.downsample_ratio)),int(exp_config.camera_image_patch_size[1]/(4*exp_config.downsample_ratio)))),pose_ref, calib_ref,pose_cur,calib_cur, cam_intrinsic,reference_frame=is_reference_sample)
    warp_trans2 = utils.argoverse_tensorflow_project_to_ground(exp_config, image,np.zeros((int(exp_config.camera_image_patch_size[0]/(8*exp_config.downsample_ratio)),int(exp_config.camera_image_patch_size[1]/(8*exp_config.downsample_ratio)))),pose_ref, calib_ref,pose_cur,calib_cur, cam_intrinsic,reference_frame=is_reference_sample)
    warp_trans3 = utils.argoverse_tensorflow_project_to_ground(exp_config, image,np.zeros((int(exp_config.camera_image_patch_size[0]/(16*exp_config.downsample_ratio)),int(exp_config.camera_image_patch_size[1]/(16*exp_config.downsample_ratio)))),pose_ref, calib_ref,pose_cur,calib_cur, cam_intrinsic,reference_frame=is_reference_sample)
    
   
    
    warped_img, warped_cover, coordinate_transform = utils.argoverse_project_to_ground(image,image_labels[...,exp_config.num_bev_classes],calib_ref,pose_ref,calib_cur,pose_cur,cam_intrinsic,reference_frame=is_reference_sample)
    if is_reference_sample:
        padded_vis_mask = np.zeros((exp_config.project_patch_size[1],exp_config.project_patch_size[0]))
        padded_vis_mask[50:-50,48:-48] = np.squeeze(vis_mask)
        warped_cover = padded_vis_mask
    # save_array(np.expand_dims(image_labels,axis=0),'pre_resize',is_rgb=False)
    new_sizes = (exp_config.camera_image_patch_size[1],exp_config.camera_image_patch_size[0])
    cropped_label = np.uint8(cv2.resize(image_labels, (int(exp_config.camera_image_patch_size[1]/4),int(exp_config.camera_image_patch_size[0]/4)), interpolation = cv2.INTER_LINEAR)>0.5)
    cropped_img = cv2.resize(image,new_sizes, interpolation = cv2.INTER_LINEAR)
    # save_array(np.expand_dims(cropped_label,axis=0),'temp_res',is_rgb=False)
    return (cropped_img, cropped_label,np.float32(warped_cover),warped_img, coordinate_transform,np.reshape(warp_trans1,[-1])[0:8],np.reshape(warp_trans2,[-1])[0:8],np.reshape(warp_trans3,[-1])[0:8],bev_labels)


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

def run_training(continue_run):

    logging.error('EXPERIMENT : ' + str(exp_config.experiment_name))
    logging.error('THIS IS : ' + str(log_dir))

    val_tokens = token_splits.VAL_LOGS
    train_tokens = token_splits.TRAIN_LOGS
    
    
    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name)



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
        # mask_tensor_shape = [n_seqs,exp_config.patch_size[1],exp_config.patch_size[0],exp_config.num_bev_classes + 1]
        
        images_placeholder = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
        
        image_labels_placeholder = tf.placeholder(tf.float32, shape=image_mask_tensor_shape, name='image_labels')
        image_objects_label_placeholder= tf.placeholder(tf.float32, shape= [1,int(exp_config.camera_image_patch_size[0]/4),int(exp_config.camera_image_patch_size[1]/4),exp_config.num_object_classes+1], name='image_object_labels')
        
        separate_covers_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,exp_config.patch_size[1],exp_config.patch_size[0],1], name='separate_covers')
        
        bev_transforms_placeholder = tf.placeholder(tf.float32, shape=[np.max([1,n_seqs-1]),8], name='bev_transforms')
        
        ground_transforms_placeholder1 = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,8], name='ground_transforms1')
        
        ground_transforms_placeholder2 = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,8], name='ground_transforms2')
        
        ground_transforms_placeholder3 = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,8], name='ground_transforms3')
        
        
        coordinate_ground_transforms_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,3,3], name='coordinate_ground_transforms')
        
       
        
        ref_bev_labels_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,exp_config.label_patch_size[1],exp_config.label_patch_size[0],exp_config.num_bev_classes +2], name='ref_bev_labels')          
        
      
    
        no_mask_tensor = tf.constant(-np.ones((1,int(exp_config.patch_size[1]/exp_config.feature_downsample),int(exp_config.patch_size[0]/exp_config.feature_downsample),int(exp_config.num_classes+1)),np.float32))
        
        '''
        Extract features from the CAMERA IMAGE
        '''
        
        image_total_backbone_out, image_total_relative_endpoints, image_total_end_points  =mem_net.image_encoder(images_placeholder,no_mask_tensor,my_model_options,downsample_stages=4,use_deeplab=use_deeplab,is_training=training_time_placeholder, reuse=False)
#            image_total_backbone_out = mem_net.my_image_decoder(image_total_relative_endpoints,image_total_backbone_out,reuse=False)
        
        
        total_input_image = image_total_backbone_out
        
        side_mask_logits,side_occ_est_logits, side_masks, side_occ_softmaxed = mem_net.compat_my_side_decoder(image_total_relative_endpoints,total_input_image,num_classes=1,reuse=False)
        
        reference_image_endpoints=[]
        for endi in range(len(image_total_relative_endpoints)):
            reference_image_endpoints.append(tf.slice(image_total_relative_endpoints[endi],[reference_frame_index,0,0,0],[1,-1,-1,-1]))
        
        side_obj_logits, side_obj_softmaxed = mem_net.my_object_side_decoder(reference_image_endpoints,tf.slice(total_input_image,[reference_frame_index,0,0,0],[1,-1,-1,-1]),exp_config,reuse=False)

        alpha_pos = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(exp_config.image_object_positive_weights,axis=0),axis=0),axis=0), tf.float32)


        side_obj_loss, _ = mem_net.classwise_object_loss(side_obj_softmaxed,image_objects_label_placeholder,tf.slice(image_labels_placeholder,[reference_frame_index,0,0,exp_config.num_classes+1],[1,-1,-1,-1]),exp_config,alpha_pos,weight=True,weight_vector=None, focal=True)

        side_seg_loss0, side_alpha0 = mem_net.argoverse_contrastive_sigmoid_loss(side_masks,image_labels_placeholder,exp_config,weight=True)
        
  
        
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
        projected_obj_estimates = projected_obj_estimates*tf.squeeze(tf.slice(separate_covers_placeholder,[0,reference_frame_index,0,0,0],[1,1,-1,-1,-1]),axis=0)
        
        combined_projected_estimates = tf.concat([combined_projected_estimates,projected_obj_estimates],axis=-1)
        
        resized_combined_projected_estimates = tf.image.resize(
                combined_projected_estimates, [int(exp_config.patch_size[1]/8),int(exp_config.patch_size[0]/8)] ,method='bilinear',name='projected_estimates_resize'  )
        
        bigger_resized_combined_projected_estimates = tf.image.resize(
                combined_projected_estimates, [int(exp_config.patch_size[1]/4),int(exp_config.patch_size[0]/4)] ,method='bilinear',name='bigger_projected_estimates_resize'  )
 
        logging.error('BIGGER PROJ ' + str(bigger_resized_combined_projected_estimates))
    
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
        
        combined_end = tf.reduce_max(all_bev_end2*cur_separate_covers,axis=0,keepdims=True)
        
        combined_end = tf.concat([combined_end,tf.slice(all_bev_end2,[reference_frame_index,0,0,0],[1,-1,-1,-1])],axis=-1)
        
        combined_end = tf.image.resize(
                combined_end, [int(exp_config.patch_size[1]/4),int(exp_config.patch_size[0]/4)] ,method='bilinear',name='projected_estimates_resize'  )
        
        bev_total_relative_endpoints = [tf.concat([combined_end,bigger_resized_combined_projected_estimates],axis=-1)]
        
        total_input = tf.concat([ resized_combined_projected_estimates,bev_total_backbone_out],axis=-1)
        
        static_logits, static_masks,object_logits, object_masks = mem_net.my_bev_object_decoder(bev_total_relative_endpoints,total_input,exp_config,reuse=False)
        
        alpha_pos = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(exp_config.bev_positive_weights,axis=0),axis=0),axis=0), tf.float32)
        alpha_neg = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(exp_config.bev_negative_weights,axis=0),axis=0),axis=0), tf.float32)

        masks = tf.concat([static_masks,object_masks],axis=-1)
        seg_loss0, alpha0 = mem_net.classwise_modified_bev_object_loss(masks,ref_bev_labels_placeholder,None,exp_config,alpha_pos,alpha_neg,weight=True)
        

        '''
        LOSSES ADDED
        '''
        mean_side_seg_loss0 = tf.reduce_mean(side_seg_loss0)
        mean_side_obj_loss = tf.reduce_mean(side_obj_loss)
        mean_seg_loss0 = tf.reduce_mean(seg_loss0)
#            mean_seg_loss1 = tf.reduce_mean(seg_loss1)
        
        occ_loss = tf.constant(0)
        
        recon_loss = mean_seg_loss0 
        
        side_loss = mean_side_seg_loss0 + 0.001*tf.reduce_mean(side_occ_loss0) + 2*mean_side_obj_loss
        l2_loss_vars = []
        trainable_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in trainable_vars:
            cur_name = var.op.name.split('/')[-1]
            if not (('bias' in cur_name) | ('_b' in cur_name) | ('gamma' in var.op.name) | ('beta' in var.op.name)):
                l2_loss_vars.append(var)
            
        logging.error('L2 vars')
        logging.error(str(l2_loss_vars))
        lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in l2_loss_vars])/len(l2_loss_vars)
        loss = recon_loss + 0.0002*lossL2 + 0.5*side_loss 

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_placeholder)

        
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        trainable_vars =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        optimizer_variables=[]

        backbone_optimizer_variables=[]
        to_load_vars = []
        
        for var in all_vars:
                    
          if 'my_bev_object_decoder' in var.op.name:
              if 'processed_endpoint_init_conv' in var.op.name:
                  continue
              
              else:    
                  to_load_vars.append(var)
          elif 'disc' in var.op.name:
            continue
          
          
        
              
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
            # bev_variables_to_restore = dict()
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

            else:
                logging.error('NON BACKBONE VAR ' + str(var))
                optimizer_variables.append(var)
        
        logging.error('NON DECODER VARS '+ str(non_decoder_vars))
        logging.error('NUMBER OF ALL PARAMETERS: ' + str(np.sum([np.prod(v.get_shape().as_list()) for v in optimizer_variables])))
        logging.error('NUMBER OF BACKBONE PARAMETERS: ' + str(np.sum([np.prod(v.get_shape().as_list()) for v in backbone_optimizer_variables])))
            
        # to_load_saver = tf.train.Saver(var_list=to_load_vars,max_to_keep=2)
        
        gvs = optimizer.compute_gradients(loss,var_list=optimizer_variables)
        
        capped_gvs = get_clipped_grads(gvs)
        
        network_train_op_total = optimizer.apply_gradients(capped_gvs)
                
        
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
#       
        else:
  
            load_path = os.path.join('/scratch_net/catweazle/cany/argoverse_classwise/logdir','checkpoints/keep2','best-16-9999')
  
            saver.restore(sess,load_path)
            # to_load_saver.restore(sess,load_path)
        sess.run(mem_net.interp_surgery(tf.global_variables()))
        
                
        init_step = 0
        
        
        time2=0
        time3 = 0
        start_epoch = 0
        step = init_step
        curr_lr = exp_config.learning_rate

        
        curr_lr = 1e-06
        

        max_epoch = 3000
        best_mean = 0.16
        
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
        
     
        
        for epoch in range(start_epoch,max_epoch):
   
            
            if epoch % 10 == 0:
                curr_lr = 0.9*curr_lr
            
            logging.error('EPOCH : ' + str(epoch))
            # Update learning rate if necessary

            random.shuffle(train_frame_list)

            generators_list=[]
  
            max_interval_between_frames = 15
            for k in range(BATCH_SIZE):
                generators_list.append( BackgroundGenerator(iterate_minibatches(train_frame_list ,max_interval_between_frames, reference_frame_index=reference_frame_index, n_frames_per_seq= n_frames_per_seq,
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
 
                                else:
                                    
                                    batch_image, batch_label, batch_bev_images,batch_transforms,batch_tf_transforms1, batch_tf_transforms2,batch_tf_transforms3,batch_separate_covers, batch_coordinate_transforms,batch_ref_bev_labels,all_bev_labels = temp_next
                             
                                    
                            if end_of_epoch:
    #                        
                                break
                       
                            if step % 5000 == 4999:
                                saver.save(sess,
                                 os.path.join(log_dir,
                                              'checkpoints',
                                              'routine'),
                                 global_step=step)
                                
                               
                            batch_bev_images = np.squeeze(batch_bev_images)
                            
                            to_feed_image_labels = np.zeros((n_frames_per_seq,int(exp_config.camera_image_patch_size[0]/4),int(exp_config.camera_image_patch_size[1]/4),total_label_slices),np.uint8)
                            to_feed_image_object_labels = np.zeros((1,int(exp_config.camera_image_patch_size[0]/4),int(exp_config.camera_image_patch_size[1]/4),exp_config.num_object_classes+1),np.uint8)
                            
                            to_feed_image_labels[...,0] = np.copy(batch_label[...,0])
                            to_feed_image_labels[...,1:] = np.copy(batch_label[...,exp_config.num_bev_classes:])
                            
                            to_feed_image_object_labels[...,:exp_config.num_object_classes] = np.copy(np.expand_dims(batch_label[reference_frame_index,...,1:exp_config.num_bev_classes],axis=0))
                            to_feed_image_object_labels[...,-1] = np.clip(1-np.sum(to_feed_image_object_labels,axis=-1),0,1)
        
                            feed_dict = {
                                    
                            learning_rate_placeholder:curr_lr,
                            training_time_placeholder: True,
                            my_training_placeholder:True,
                           
                            images_placeholder:batch_image,
                            
                            image_labels_placeholder:to_feed_image_labels,
                            image_objects_label_placeholder:to_feed_image_object_labels,
                            bev_transforms_placeholder:batch_transforms,
                            separate_covers_placeholder : batch_separate_covers,
                            ground_transforms_placeholder1:batch_tf_transforms1,
                            ground_transforms_placeholder2:batch_tf_transforms2,
                            ground_transforms_placeholder3:batch_tf_transforms3,
                            coordinate_ground_transforms_placeholder:batch_coordinate_transforms,
                            
                          
                            ref_bev_labels_placeholder: batch_ref_bev_labels
                            
                            
                            }
                            
                          
                            _ = sess.run(network_train_op_total, feed_dict=feed_dict)
                                
    
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
                                
                          
                            if step % exp_config.val_eval_frequency == (exp_config.val_eval_frequency - 1):
    
                                  val_res=do_eval(sess,val_tokens,
                                                      my_training_placeholder,
                                
                                  images_placeholder,
                              
                                  image_labels_placeholder,
                                 
                                  bev_transforms_placeholder,
                                  separate_covers_placeholder,
                                  ground_transforms_placeholder1,
                                  ground_transforms_placeholder2,
                                  ground_transforms_placeholder3,
                                  coordinate_ground_transforms_placeholder,
                                
                             
                                projected_obj_estimates,
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
                  
                    logging.error('MAIN LOOP EXCEPTION ' + str(e))
                    
                    continue
   


def standard_iterate_minibatches(my_scene, chosen_frame , max_interval_between_frames,
                                                                          reference_frame_index,
                                                                          
                                                                         n_frames_per_seq=3,
                                                                         batch_size=1): 

    n_seqs = n_frames_per_seq-num_frames+1
    try:
        
        
        camera = "ring_front_center"
        scene = train_loader.get(my_scene)
        
        pool = ThreadPool(n_seqs*num_frames) 
        
        n_frames_in_scene = scene.num_lidar_frame
        
        # logging.error('NUM FRAMES ' + str(n_frames_in_scene))
        # logging.error('NUM FRAMES ' + str(n_frames_in_scene))
        seq_length = np.copy(n_frames_in_scene)
        frame_ids=[]
        # logging.error('FIRST FRAME TO GET')
        first_frame = random.randint(np.max([0, chosen_frame - max_interval_between_frames]),chosen_frame)
        # logging.error('FIRST FRAME GOT')
        frame_ids.append(first_frame)
        frame_ids.append(chosen_frame)
        last_frame = random.randint(np.min([seq_length - 1, chosen_frame]),np.min([chosen_frame + max_interval_between_frames,seq_length - 1]) )
        frame_ids.append(last_frame)
       
        pairs = []
        
      
        timestamp = str(np.copy(train_loader._image_timestamp_list_sync[my_scene][camera][frame_ids[reference_frame_index]]))
        # logging.error('TIMESTAMP GET')
        # logging.error('TIME S ' + timestamp)
        output_path = os.path.join(exp_config.argo_labels_path,
                                       my_scene, camera, 
                                       str(camera)+'_'+str(timestamp)+'.png')
        
        encoded_labels = np.flipud(np.array(Image.open(output_path),np.int32))
        # logging.error('ENCODED LABELS SHAPE ' + str(encoded_labels.shape))
        num_class = exp_config.num_bev_classes
        bev_labels = decode_binary_labels(encoded_labels, num_class+ 1)
        # mask = ~labels[...,-1]
        
        # labels = labels[...,:-1]
        calib_cur = train_loader.get_calibration(camera, my_scene)
        calib_ref =calib_cur
        to_image_transform = utils.argoverse_project_to_image(exp_config, np.zeros_like(bev_labels),calib_ref)

        # image_objects = cv2.warpPerspective(np.squeeze(bev_labels[...,-1]),to_image_transform,exp_config.original_image_size,flags=cv2.INTER_LINEAR)
        # image_objects= cv2.resize(image_objects,(int(exp_config.camera_image_patch_size[1]/4),int(exp_config.camera_image_patch_size[0]/4)), interpolation = cv2.INTER_LINEAR)
        # vis_mask = np.float32(image_objects > 0.5)
        vis_mask = np.copy(np.uint8(np.flipud(utils.get_visible_mask(calib_cur.K, calib_cur.camera_config.img_width,
                                   exp_config.map_extents, exp_config.resolution))))
        
        # logging.error('VIS MASK SHAPE ' + str(vis_mask.shape))
        # logging.error('BEV LABELS SHAPE ' + str(bev_labels.shape))        
        vis_mask = np.expand_dims(vis_mask,axis=-1)
        
      
        bev_labels = np.concatenate([np.copy(bev_labels[...,:exp_config.num_bev_classes]),np.copy(vis_mask),vis_mask*(1-np.copy(np.expand_dims(bev_labels[...,exp_config.num_bev_classes],axis=-1)))],axis=-1)
      
        # logging.error('ABOUT TO GET POSE')
        pose_ref = np.copy(scene.get_pose(frame_ids[reference_frame_index]).transform_matrix)
        # logging.error('GOT POSE')
        for k in range(n_seqs):
            for m in range(num_frames):
                timestamp = str(np.copy(train_loader._image_timestamp_list_sync[my_scene][camera][frame_ids[m]]))
                # logging.error('TIME ' + str(m) + ' ' + timestamp)
                pose_cur = np.copy(scene.get_pose(frame_ids[m]).transform_matrix)
                output_path = os.path.join(exp_config.argo_labels_path, 
                                           my_scene, camera, 
                                           str(camera)+'_'+str(timestamp)+'.png')
                
                image_string = os.path.join(exp_config.argo_track_path,my_scene,'ring_front_center','ring_front_center_'+str(timestamp)+'.jpg')
    
                
                pairs.append((image_string,output_path,vis_mask,calib_cur,pose_ref,pose_cur,m==reference_frame_index))
        
   
        # logging.error('GIVEN TO POOL')
        results = pool.map(single_process,pairs)
        
        
        pool.close() 
        pool.join() 
#        logging.error('Results shape : ' + str(len(results)))

        seq_images_ar=np.zeros((n_frames_per_seq,exp_config.camera_image_patch_size[0],exp_config.camera_image_patch_size[1],3),np.float32)
        seq_labels_ar=np.ones((n_frames_per_seq,int(exp_config.camera_image_patch_size[0]/4),int(exp_config.camera_image_patch_size[1]/4),exp_config.num_bev_classes+2),np.float32)

        bev_transforms_ar1=np.ones((n_seqs,num_frames,8),np.float32)
        bev_transforms_ar2=np.ones((n_seqs,num_frames,8),np.float32)
        bev_transforms_ar3=np.ones((n_seqs,num_frames,8),np.float32)
        coordinate_transforms_ar=np.ones((n_seqs,num_frames,3,3),np.float32)
        bev_images_ar=np.zeros((n_seqs,num_frames,exp_config.patch_size[1],exp_config.patch_size[0],3),np.float32)
        
        bev_covers_ar=np.ones((n_seqs,num_frames,exp_config.patch_size[1],exp_config.patch_size[0],1),np.float32)

        all_bev_labels_ar = np.zeros((num_frames,196,200,exp_config.num_bev_classes+2),np.float32)

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
            bev_images_ar[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[3])
            
            bev_covers_ar[int(k//num_frames),k%num_frames,...] = np.expand_dims(np.copy(temp_res[2]),axis=-1)
            
            bev_transforms_ar1[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[5])
            bev_transforms_ar2[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[6])
            bev_transforms_ar3[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[7])
            coordinate_transforms_ar[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[4])
            all_bev_labels_ar[k,...]= np.copy(temp_res[8])
        
        return seq_images_ar, seq_labels_ar,bev_images_ar,bev_covers_ar,np.zeros((1,8)), bev_transforms_ar1,bev_transforms_ar2,bev_transforms_ar3,coordinate_transforms_ar,np.expand_dims(bev_labels,axis=0),to_image_transform,all_bev_labels_ar,True

    except Exception as e:
        pool.close() 
        pool.join() 
        logging.error('ITERATE MINI BATCHES Exception ' + str(e))
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # logging.error(str(exc_type, fname, exc_tb.tb_lineno))
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
        
        
        cur_frame = train_tokens[b_i]
        
        indi = all_frames_cumulative > cur_frame
        selected_seq = np.min(np.where(indi)[0])
        
        
        temp_sum = np.sum(all_frames_list[:selected_seq])
        
        frame_id = int(cur_frame - temp_sum)
        

        
        my_token = token_splits.TRAIN_LOGS[selected_seq]

        seq_images_ar,  seq_labels_ar, bev_images_ar,bev_covers_ar,transforms_ar,tf_transforms1,tf_transforms2,tf_transforms3, coordinate_transforms_ar,real_ref_bev_labels,to_image_transform,all_bev_labels_ar,went_well =  standard_iterate_minibatches(my_token,frame_id,max_interval_between_frames,
                                                                         
                                                                         reference_frame_index,
                                                                         n_frames_per_seq=n_frames_per_seq,
                                                                         batch_size=batch_size
                                                                         )
        
        
        if not went_well:
            continue
       
        
        yield seq_images_ar,seq_labels_ar,bev_images_ar, transforms_ar,tf_transforms1,tf_transforms2,tf_transforms3, bev_covers_ar, coordinate_transforms_ar,real_ref_bev_labels,all_bev_labels_ar

def eval_iterator(my_scene,cur_index,single_frame=False): 

    n_seqs = 1
  
        
    
    camera = "ring_front_center"
    scene = train_loader.get(my_scene)
    
    pool = ThreadPool(n_seqs*num_frames) 
    
    n_frames_in_scene = scene.num_lidar_frame
    # logging.error('NUM FRAMES ' + str(n_frames_in_scene))
    seq_length = n_frames_in_scene
    frame_ids=[]
    first_frame = cur_index

    frame_ids.append(first_frame)
    
    # logging.error('LEN IMAGES ' + str(len(all_images_list)))
    
    if single_frame:
    
        for frame_number in range(1,n_frames_per_seq):
            frame_ids.append(first_frame )
    else:
        
        for frame_number in range(1,n_frames_per_seq):
            frame_ids.append(first_frame + frame_number)
        
    pairs = []
    

    timestamp = str(np.copy(train_loader._image_timestamp_list_sync[my_scene][camera][frame_ids[reference_frame_index]]))

    # logging.error('TIME S ' + timestamp)
    output_path = os.path.join(exp_config.argo_labels_path,
                                   my_scene, camera, 
                                   str(camera)+'_'+str(timestamp)+'.png')
    
    encoded_labels = np.flipud(np.array(Image.open(output_path),np.int32))
    # logging.error('ENCODED LABELS SHAPE ' + str(encoded_labels.shape))
    num_class = exp_config.num_bev_classes
    bev_labels = decode_binary_labels(encoded_labels, num_class+ 1)
    # mask = ~labels[...,-1]
    
    # labels = labels[...,:-1]
    calib_cur = train_loader.get_calibration(camera, my_scene)
    calib_ref =calib_cur
    to_image_transform = utils.argoverse_project_to_image(exp_config, np.zeros_like(bev_labels),calib_ref)

    # image_objects = cv2.warpPerspective(np.squeeze(bev_labels[...,-1]),to_image_transform,exp_config.original_image_size,flags=cv2.INTER_LINEAR)
    # image_objects= cv2.resize(image_objects,(int(exp_config.camera_image_patch_size[1]/4),int(exp_config.camera_image_patch_size[0]/4)), interpolation = cv2.INTER_LINEAR)
    # vis_mask = np.float32(image_objects > 0.5)
    vis_mask = np.copy(np.uint8(np.flipud(utils.get_visible_mask(calib_cur.K, calib_cur.camera_config.img_width,
                               exp_config.map_extents, exp_config.resolution))))
    
    # logging.error('VIS MASK SHAPE ' + str(vis_mask.shape))
    # logging.error('BEV LABELS SHAPE ' + str(bev_labels.shape))        
    vis_mask = np.expand_dims(vis_mask,axis=-1)
    bev_labels = np.concatenate([bev_labels[...,:exp_config.num_bev_classes],vis_mask,vis_mask*(1-np.expand_dims(bev_labels[...,exp_config.num_bev_classes],axis=-1))],axis=-1)
    
    
    pose_ref = np.copy(scene.get_pose(frame_ids[reference_frame_index]).transform_matrix)
    for k in range(n_seqs):
        for m in range(num_frames):
            timestamp = str(np.copy(train_loader._image_timestamp_list_sync[my_scene][camera][frame_ids[m]]))
            # logging.error('TIME ' + str(m) + ' ' + timestamp)
            pose_cur = np.copy(scene.get_pose(frame_ids[m]).transform_matrix)
            output_path = os.path.join(exp_config.argo_labels_path, 
                                       my_scene, camera, 
                                       str(camera)+'_'+str(timestamp)+'.png')
            
            image_string = os.path.join(exp_config.argo_track_path,my_scene,'ring_front_center','ring_front_center_'+str(timestamp)+'.jpg')

            
            pairs.append((image_string,output_path,vis_mask,calib_cur,pose_ref,pose_cur,m==reference_frame_index))
    
   

    results = pool.map(single_process,pairs)
    
    
    pool.close() 
    pool.join() 
#        logging.error('Results shape : ' + str(len(results)))

    seq_images_ar=np.zeros((n_frames_per_seq,exp_config.camera_image_patch_size[0],exp_config.camera_image_patch_size[1],3),np.float32)
    seq_labels_ar=np.ones((n_frames_per_seq,int(exp_config.camera_image_patch_size[0]/4),int(exp_config.camera_image_patch_size[1]/4),exp_config.num_bev_classes+2),np.float32)

    bev_transforms_ar1=np.ones((n_seqs,num_frames,8),np.float32)
    bev_transforms_ar2=np.ones((n_seqs,num_frames,8),np.float32)
    bev_transforms_ar3=np.ones((n_seqs,num_frames,8),np.float32)
    coordinate_transforms_ar=np.ones((n_seqs,num_frames,3,3),np.float32)
    bev_images_ar=np.zeros((n_seqs,num_frames,exp_config.patch_size[1],exp_config.patch_size[0],3),np.float32)
    
    bev_covers_ar=np.ones((n_seqs,num_frames,exp_config.patch_size[1],exp_config.patch_size[0],1),np.float32)

    all_bev_labels_ar = np.zeros((num_frames,196,200,exp_config.num_bev_classes+2),np.float32)

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
        bev_images_ar[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[3])
        
        bev_covers_ar[int(k//num_frames),k%num_frames,...] = np.expand_dims(np.copy(temp_res[2]),axis=-1)
        
        bev_transforms_ar1[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[5])
        bev_transforms_ar2[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[6])
        bev_transforms_ar3[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[7])
        coordinate_transforms_ar[int(k//num_frames),k%num_frames,...] = np.copy(temp_res[4])
        all_bev_labels_ar[k,...]= np.copy(temp_res[8])
    
    return seq_images_ar, seq_labels_ar,bev_images_ar,bev_covers_ar,np.zeros((1,8)), bev_transforms_ar1,bev_transforms_ar2,bev_transforms_ar3,coordinate_transforms_ar,np.expand_dims(bev_labels,axis=0),to_image_transform,all_bev_labels_ar,True

def overall_eval_iterator(my_scene,cur_index, single_frame=False):
    
    # logging.error('SINGLE FRAME ' + str(single_frame))
    
    
    seq_images_ar, seq_labels_ar,bev_images_ar,bev_covers_ar,_, tf_transforms1,tf_transforms2,tf_transforms3,coordinate_transforms_ar,real_ref_bev_labels,to_image_transform,all_bev_labels_ar,went_well =eval_iterator(my_scene,cur_index,single_frame=single_frame)
     

    squeezed_bev_covers_ar = np.squeeze(bev_covers_ar,axis=-1)
    
    total_img_list=[]
    # total_labels_list=[]

    for k in range(n_seqs):
        total_img = np.zeros_like(bev_images_ar[0,0,...])
        
        for m in range(num_frames):
            total_img[squeezed_bev_covers_ar[k,m,...]>0.5,:] = bev_images_ar[k,m,...][squeezed_bev_covers_ar[k,m,...]>0.5,:]
            # total_labels[squeezed_bev_covers_ar[k,m,...]>0.5,:] = bev_labels_ar[k,m,...][squeezed_bev_covers_ar[k,m,...]>0.5,:]
            
        total_img_list.append(total_img)
        # total_labels_list.append(total_labels)
    
    fin_bev_images = np.stack(total_img_list,axis=0)
    # fin_bev_labels = np.stack(total_labels_list,axis=0)
    fin_covers = np.clip(np.sum(bev_covers_ar,axis=1),0,1)
    
    my_area = np.float32(bev_covers_ar > 0.5)
    
#        logging.error('BEV IMAGES MAX ' + str(np.max(bev_images_ar))+ ' MIN ' + str(np.min(bev_images_ar)))
    
    separate_bev_images = my_area*bev_images_ar
    
    to_return_bev_images_list = []
    
    for k in range(num_frames):
        to_return_bev_images_list.append(separate_bev_images[:,k,...])
        
    to_return_bev_images = np.concatenate(to_return_bev_images_list,axis=-1)
   
    return seq_images_ar, seq_labels_ar, fin_bev_images,fin_covers , tf_transforms1,tf_transforms2,tf_transforms3, bev_covers_ar, coordinate_transforms_ar,to_return_bev_images,real_ref_bev_labels




def do_eval(sess,val_tokens,
                                            my_training_placeholder,
                        
                        images_placeholder,
                        
                        image_labels_placeholder,
                        bev_transforms_placeholder,
                        
                        separate_covers_placeholder,
                        ground_transforms_placeholder1,
                        ground_transforms_placeholder2,
                        ground_transforms_placeholder3,
                        coordinate_ground_transforms_placeholder,
                        
                        projected_obj_estimates,
                        masks,
                        side_masks,side_occ_masks,side_obj_softmaxed,
                        projected_estimates,
                       
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
    
    
    for my_scene_token in val_tokens:
        
        scene_results=[]
        occ_scene_results=[]
        # current_dir = os.path.join(target_dir,'scene'+my_scene_token)
        
        # images = sorted(glob.glob(os.path.join(current_dir,'img*.png')))
        # labels = sorted(glob.glob(os.path.join(current_dir,'label*.png')))
        name_of_seq = my_scene_token
        
        # scene_token = current_dir.split('/')[-1][5:]
        
        # my_scene = nusc.get('scene', scene_token)
        scene = train_loader.get(my_scene_token)
        # logging.error('SCENE ' + my_scene_token)
        n_frames_in_scene = scene.num_lidar_frame
        if not os.path.exists(os.path.join(validation_res_path,name_of_seq)):
            os.makedirs(os.path.join(validation_res_path,name_of_seq))
        
        
        
        for frame_number in range(0,n_frames_in_scene,10):
#            logging.error('FRAME NUMBER ' + str(frame_number))
            
            if frame_number < reference_frame_index:
                
                
                batch_image, batch_label, batch_bev_images, batch_bev_covers ,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels =\
                overall_eval_iterator(my_scene_token,frame_number,single_frame=True)
            elif (frame_number >= (n_frames_in_scene - (num_frames - reference_frame_index - 1))):
            
                batch_image, batch_label, batch_bev_images, batch_bev_covers ,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels =\
                overall_eval_iterator(my_scene_token,frame_number,single_frame=True)
                
            else:
                
                batch_image, batch_label, batch_bev_images, batch_bev_covers ,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels =\
                overall_eval_iterator(my_scene_token,int(frame_number - reference_frame_index),single_frame=False)
                
            
            
            # logging.error('WE GOT THE STUFF')
            feed_dict = {
                                
           
            training_time_placeholder: False,
            my_training_placeholder:False,
            
            images_placeholder:batch_image,
              
            bev_transforms_placeholder:np.zeros((1,8)),
          
            separate_covers_placeholder : batch_separate_covers,
            ground_transforms_placeholder1:batch_tf_transforms1,
            ground_transforms_placeholder2:batch_tf_transforms2,
            ground_transforms_placeholder3:batch_tf_transforms3,
            coordinate_ground_transforms_placeholder:batch_coordinate_transforms,
            
            
            }

            masks_v,side_masks_v,side_occ_masks_v, projected_estimates_v,combined_projected_estimates_v,side_obj_softmaxed_v, projected_obj_v = sess.run([masks,
                        side_masks,side_occ_masks,projected_estimates,
                        combined_projected_estimates,side_obj_softmaxed,projected_obj_estimates], feed_dict=feed_dict)
#            logging.error('DEC OUTPUT ' + str(temp_dec_output.shape))
            
            
            # hard_estimate_list = []
            sample_results=[]
            squeezed_masks = np.squeeze(masks_v)
           
            hard_estimates = np.uint8(squeezed_masks[...,:exp_config.num_bev_classes] > 0.45)
            
            
            for k in range(exp_config.num_bev_classes):
                # bev_estimate = np.squeeze(masks_v[...,k])
                # hard_estimate_list.append(np.uint8(bev_estimate > 0.5))
                all_stats , void_pixels= utils.get_all_stats(np.squeeze(batch_ref_bev_labels[...,k]), np.uint8(hard_estimates[...,k]),np.squeeze(batch_ref_bev_labels[...,exp_config.num_bev_classes+1]),mask_iou=exp_config.use_occlusion)
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
    union = tot_j1[...,1]
    tp = tot_j1[...,2]
    fp = tot_j1[...,3]
    fn = tot_j1[...,4]
    tn = tot_j1[...,5]
    gt_exists = tot_j1[...,-1]
    
    
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
    write_to_txt_file(os.path.join(log_dir,'val_results.txt'),res_strings)
    return confuse_iou


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
