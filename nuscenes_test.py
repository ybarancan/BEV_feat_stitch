
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


import glob

from PIL import Image

from deeplab import common

import mem_net

from multiprocessing.dummy import Pool as ThreadPool 

import utils

from nuscenes.nuscenes import NuScenes

from nuscenes.map_expansion.map_api import NuScenesMap

from dataset import token_splits

from experiments import nuscenes_objects_base_val as exp_config

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
frame_interval = 3
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

    
    logging.error('EXPERIMENT : ' + str(exp_config.experiment_name))
    logging.error('THIS IS : ' + str(log_dir))
   
    
    val_tokens = token_splits.VAL_SCENES
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
#        with tf.Graph().as_default():
    with tf.Session(config = config) as sess:
        # Generate placeholders for the images and labels.



        training_time_placeholder = tf.placeholder(tf.bool, shape=[])
        
        my_training_placeholder = tf.placeholder(tf.bool, shape=[])
        
        # Build a Graph that computes predictions from the inference model.
        my_model_options = common.ModelOptions({common.OUTPUT_TYPE:10},crop_size=exp_config.camera_image_patch_size,atrous_rates=[6, 12, 18])
   
        image_tensor_shape = [n_frames_per_seq,exp_config.camera_image_patch_size[0],exp_config.camera_image_patch_size[1],3]
        image_mask_tensor_shape = [n_frames_per_seq,int(exp_config.camera_image_patch_size[0]/4),int(exp_config.camera_image_patch_size[1]/4),total_label_slices]
        
        images_placeholder = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
        
        image_labels_placeholder = tf.placeholder(tf.float32, shape=image_mask_tensor_shape, name='image_labels')
        
        covers_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,exp_config.patch_size[1],exp_config.patch_size[0],1], name='covers')
        
        separate_covers_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,exp_config.patch_size[1],exp_config.patch_size[0],1], name='separate_covers')
        
        bev_transforms_placeholder = tf.placeholder(tf.float32, shape=[np.max([1,n_seqs-1]),8], name='bev_transforms')
        
        ground_transforms_placeholder1 = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,8], name='ground_transforms1')
        
        ground_transforms_placeholder2 = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,8], name='ground_transforms2')
        
        ground_transforms_placeholder3 = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,8], name='ground_transforms3')
        
        
        coordinate_ground_transforms_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,3,3], name='coordinate_ground_transforms')
        
        channel_bev_images_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,exp_config.patch_size[1],exp_config.patch_size[0],3*num_frames], name='channel_images')
        
     
        no_mask_tensor = tf.constant(-np.ones((1,int(exp_config.patch_size[1]/exp_config.feature_downsample),int(exp_config.patch_size[0]/exp_config.feature_downsample),int(exp_config.num_classes+1)),np.float32))
        
        '''
        Extract features from the CAMERA IMAGE
        '''
        
        image_total_backbone_out, image_total_relative_endpoints, image_total_end_points  =mem_net.image_encoder(images_placeholder,no_mask_tensor,my_model_options,downsample_stages=4,use_deeplab=use_deeplab,is_training=training_time_placeholder, reuse=False)

        total_input_image = image_total_backbone_out
        
        side_mask_logits,side_occ_est_logits, side_masks, side_occ_softmaxed = mem_net.my_side_decoder(image_total_relative_endpoints,total_input_image,reuse=False)
        
        reference_image_endpoints=[]
        for endi in range(len(image_total_relative_endpoints)):
            reference_image_endpoints.append(tf.slice(image_total_relative_endpoints[endi],[reference_frame_index,0,0,0],[1,-1,-1,-1]))
        
        side_obj_logits, side_obj_softmaxed = mem_net.my_object_side_decoder(reference_image_endpoints,tf.slice(total_input_image,[reference_frame_index,0,0,0],[1,-1,-1,-1]),exp_config,apply_softmax=True,reuse=False)
        
      
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
        
        bev_total_relative_endpoints = [combined_end]
          
        total_input = tf.concat([ resized_combined_projected_estimates,bev_total_backbone_out],axis=-1)
        
        static_logits, static_masks,object_logits, object_masks = mem_net.my_bev_object_decoder(bev_total_relative_endpoints,total_input,exp_config,apply_softmax=True,reuse=False)
        masks = tf.concat([static_masks,object_masks],axis=-1)
        
        saver = tf.train.Saver(max_to_keep=2)
        
        # saver_best_loss = tf.train.Saver(max_to_keep=2)
        init = tf.global_variables_initializer()
        sess.run(init)
        
        load_path = exp_config.load_path




        saver.restore(sess,load_path)

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


def eval_iterator(my_scene,cur_index, reference_frame_index, single_frame=False,apply_interval=False): 
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
    
    reference_samples = []
    
    bev_labels_list=[]
    
    for k in range(n_seqs):
        cur_ref_sample = nusc.sample[all_sample_inds[frame_ids[k+reference_frame_index]]]
        reference_samples.append(cur_ref_sample)
        
        cam_token_cur = cur_ref_sample['data']['CAM_FRONT']
        cam_record_cur = nusc.get('sample_data', cam_token_cur)
        
        bev_label = np.array(Image.open( os.path.join(exp_config.nuscenes_bev_root,  
                                   cam_record_cur['token'] + '.png')),np.int32)
        
        bev_label = np.flipud(bev_label)
        
        bev_label = decode_binary_labels(bev_label,exp_config.num_bev_classes+1)

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


def overall_eval_iterator(my_scene,cur_index, reference_frame_index, single_frame=False,apply_interval=False):
    
    # logging.error('SINGLE FRAME ' + str(single_frame))
    seq_images_ar, seq_labels_ar, bev_images_ar,bev_labels_ar,bev_covers_ar, transforms_ar,tf_transforms1,tf_transforms2,tf_transforms3, coordinate_transforms_ar,real_ref_bev_labels,went_well =  eval_iterator(my_scene,cur_index,reference_frame_index,single_frame=single_frame,apply_interval=apply_interval)
        

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
                       projected_obj_estimates,
                        combined_projected_estimates,
                        iteration,training_time_placeholder,val_folder_path=validation_res_path):


    logging.error('Started evaluation')
    
    
    res_strings=[]
    all_static_j1s=[]
    all_object_j1s=[]

    counter = 0
    
    runtimes = []
    
    for my_scene_token in val_tokens:
  
        counter = counter + 1
        logging.error('SCENE ' + str(counter) + ' of ' + str(len(val_tokens)) + ' scenes')
        
        scene_static_results=[]
        scene_object_results=[]
        current_dir = os.path.join(target_dir,'scene'+my_scene_token)

        images = sorted(glob.glob(os.path.join(current_dir,'img*.png')))

        name_of_seq = my_scene_token
        
        
        
        if not os.path.exists(os.path.join(validation_res_path,name_of_seq)):
            os.makedirs(os.path.join(validation_res_path,name_of_seq))
        
        
        
        for frame_number in range(len(images)):
            logging.error('FRAME NUMBER ' + str(frame_number))
            
            
            if frame_number < reference_frame_index:
                
            
                batch_image, batch_label, batch_bev_images, batch_bev_labels,batch_bev_covers ,batch_transforms,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels =\
                overall_eval_iterator(my_scene_token,frame_number,reference_frame_index,single_frame=True,apply_interval=False)
                
            elif frame_number < frame_interval*reference_frame_index:
                
                batch_image, batch_label, batch_bev_images, batch_bev_labels,batch_bev_covers ,batch_transforms,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels=\
                overall_eval_iterator(my_scene_token,int(frame_number - reference_frame_index),reference_frame_index,single_frame=False,apply_interval=False)
                
            elif (frame_number >= (len(images) - (num_frames - reference_frame_index - 1))):
            
                batch_image, batch_label, batch_bev_images, batch_bev_labels,batch_bev_covers ,batch_transforms,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels=\
                overall_eval_iterator(my_scene_token,frame_number,reference_frame_index,single_frame=True,apply_interval=False)
                
            elif (frame_number >= (len(images) - frame_interval*(num_frames - reference_frame_index - 1))):
            
                batch_image, batch_label, batch_bev_images, batch_bev_labels,batch_bev_covers ,batch_transforms,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels =\
                overall_eval_iterator(my_scene_token,int(frame_number - reference_frame_index),reference_frame_index,single_frame=False,apply_interval=False)
                    
            else:
                
                batch_image, batch_label, batch_bev_images, batch_bev_labels,batch_bev_covers ,batch_transforms,batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_channel_bev_images,batch_ref_bev_labels=\
                overall_eval_iterator(my_scene_token,int(frame_number - frame_interval*reference_frame_index),reference_frame_index,single_frame=False,apply_interval=True)
                

        
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
            
            time1 = time.time()
            masks_v= sess.run(masks, feed_dict=feed_dict)
            time2 = time.time()
            rt = time2-time1
            runtimes.append(rt)
#           
            sample_results=[]
            squeezed_masks = np.squeeze(masks_v)
            
            temp_object_estimates = squeezed_masks[...,exp_config.num_static_classes:]
            

            
            thresh_list=[0.5,0.5,0.45,0.45,
                         0.5,0.5,0.5,0.3,
                         0.3,0.6,0.45,0.45,
                         0.45,0.5]
            
            
            
            static_estimates = np.zeros((temp_object_estimates.shape[0],temp_object_estimates.shape[1],exp_config.num_static_classes))
            
            for k in range(exp_config.num_static_classes):
                static_estimates[...,k] = np.uint8(squeezed_masks[...,k] > thresh_list[k])
           
            
            object_estimates = np.zeros((temp_object_estimates.shape[0],temp_object_estimates.shape[1],exp_config.num_object_classes))
            for k in range(exp_config.num_object_classes):
                object_estimates[...,k] = np.uint8(temp_object_estimates[...,k] > thresh_list[k+4])
            
            bg_estimate = 1 - np.clip(np.sum(object_estimates,axis=-1,keepdims=True),0,1)
            
            hard_estimates = np.concatenate([static_estimates,object_estimates],axis=-1)
            
            object_estimates = np.concatenate([object_estimates,bg_estimate],axis=-1)
            
            object_stats = utils.get_confusion(exp_config, np.squeeze(batch_ref_bev_labels[...,exp_config.num_static_classes:]), object_estimates,np.squeeze(batch_ref_bev_labels[...,exp_config.num_bev_classes]),mask_iou=exp_config.use_occlusion)
             
            
            
            for k in range(exp_config.num_static_classes):
         
                all_stats , void_pixels= utils.get_all_stats(np.squeeze(batch_ref_bev_labels[...,k]), hard_estimates[...,k],np.squeeze(batch_ref_bev_labels[...,exp_config.num_bev_classes]),mask_iou=exp_config.use_occlusion)
                sample_results.append(all_stats)
                
            scene_static_results.append(np.array(sample_results))
            scene_object_results.append(object_stats)
            # occ_scene_results.append(occ_all_stats)
            
            
        
        seq_static_j1 = np.array(scene_static_results)
        seq_object_j1 = np.array(scene_object_results)
        
         
        all_static_j1s.append(np.squeeze(seq_static_j1))
        all_object_j1s.append(np.squeeze(seq_object_j1))
     
        temp_res = seq_static_j1[...,2]/(seq_static_j1[...,2]+seq_static_j1[...,3]+seq_static_j1[...,4]+0.0001)
        
        temp_string = "Iteration : " + str(iteration) + " : Scene " + str(my_scene_token)+ " - j1: " + str(np.mean(temp_res,axis=0)) 

        res_strings.append(temp_string)
        logging.error(temp_string)
        utils.write_to_txt_file(os.path.join(log_dir,'val_results.txt'),[temp_string])
    
  
    tot_static_j1 = np.concatenate(all_static_j1s,axis=0)
    tot_object_j1 = np.concatenate(all_object_j1s,axis=0)

    tp = tot_static_j1[...,2]
    fp = tot_static_j1[...,3]
    fn = tot_static_j1[...,4]
    tn = tot_static_j1[...,5]
    
    tp_rate = np.sum(tp,axis=0)/(np.sum(tp,axis=0) +  np.sum(fn,axis=0) + 0.0001)
    fp_rate = np.sum(fp,axis=0)/( np.sum(fp,axis=0) + np.sum(tn,axis=0) + 0.0001)
    
    tp_rate = np.sum(tp,axis=0)/(np.sum(tp,axis=0) +  np.sum(fn,axis=0) + 0.0001)
    fp_rate = np.sum(fp,axis=0)/( np.sum(fp,axis=0) + np.sum(tn,axis=0) + 0.0001)
    precision = np.sum(tp,axis=0)/(np.sum(tp,axis=0) +  np.sum(fp,axis=0) + 0.0001)
    
    # take_all_j = np.mean(j,axis=0)
    confuse_iou = np.sum(tp,axis=0)/(np.sum(tp,axis=0) + np.sum(fp,axis=0) + np.sum(fn,axis=0) + 0.0001)
    
    
    object_raw = np.sum(tot_object_j1,axis=0)
    
    
    conf = np.array(object_raw)
    fps = np.sum(conf,axis=-1)
    fns = np.sum(conf,axis=0)
    
    ious = []
    for k in range(10):
        ious.append(conf[k,k]/(fps[k] + fns[k] - conf[k,k]))
        
    
    temp_string = 'Static j : ' + str(confuse_iou) + '\n' +\
    ' Static tp_rate : ' + str(tp_rate)+ '\n' +' Static fp_rate : ' + str(fp_rate)+ '\n'+ ' Static precision : ' + str(precision)+ '\n'+'Object confuse : ' + str(object_raw)
    
    temp_string = temp_string + '\n' + 'Object IOU ' + str(ious)
    
    logging.error(temp_string)
    res_strings.append(temp_string)
    utils.write_to_txt_file(os.path.join(log_dir,'val_results.txt'),res_strings)
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
