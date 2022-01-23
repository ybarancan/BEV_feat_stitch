
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

from PIL import Image

from deeplab import common

import utils
import mem_net

from multiprocessing.dummy import Pool as ThreadPool 

from argoverse.data_loading.argoverse_tracking_loader \
    import ArgoverseTrackingLoader
import argoverse_token_splits as token_splits

from experiments import argoverse_objects_val_exp as exp_config


means_image = np.array([123.68, 116.779, 103.939], dtype=np.single)

total_label_slices = exp_config.num_classes + 2


train_path = os.path.expandvars(exp_config.argo_track_path)
train_loader = ArgoverseTrackingLoader(train_path)

target_dir = exp_config.argo_labels_path

exp_config.batch_size=1


use_deeplab = True
starting_from_cityscapes =False
starting_from_imagenet =False

frame_interval = exp_config.frame_interval
num_frames=exp_config.num_frames
single_frame_experiment=exp_config.single_frame_experiment

reference_frame_index = exp_config.reference_frame_index

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

logging.error('REFERENCE FRAME ' + str(reference_frame_index))

log_dir = exp_config.log_dir

train_results_path = os.path.join(log_dir,'train_results')
#log_dir = os.path.join('/raid/cany/mapmaker/logdir/', exp_config.experiment_name)
validation_res_path = os.path.join(log_dir,'val_results')

if not os.path.exists(train_results_path):
    os.makedirs(train_results_path, exist_ok=True)
    
if not os.path.exists(validation_res_path):
    os.makedirs(validation_res_path, exist_ok=True)
    

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
    # vis_mask[:-100,:] = 0
    bev_labels = np.concatenate([np.copy(bev_labels[...,:exp_config.num_bev_classes]),np.copy(vis_mask),vis_mask*(1-np.copy(np.expand_dims(bev_labels[...,exp_config.num_bev_classes],axis=-1)))],axis=-1)
    
    to_image_transform = utils.project_to_image(exp_config, np.zeros_like(bev_labels),calib_ref)

    image_labels = cv2.warpPerspective(bev_labels,to_image_transform,exp_config.original_image_size,flags=cv2.INTER_NEAREST)
    # image_objects= cv2.resize(image_objects,(int(exp_config.camera_image_patch_size[1]/4),int(exp_config.camera_image_patch_size[0]/4)), interpolation = cv2.INTER_LINEAR)
    image_labels = np.uint8(image_labels > 0.3)
    
    
    image=np.array(img, dtype=np.uint8)

    
    warp_trans1 = utils.tensorflow_project_to_ground(image,np.zeros((int(exp_config.camera_image_patch_size[0]/(4*exp_config.downsample_ratio)),int(exp_config.camera_image_patch_size[1]/(4*exp_config.downsample_ratio)))),pose_ref, calib_ref,pose_cur,calib_cur, cam_intrinsic,reference_frame=is_reference_sample)
    warp_trans2 = utils.tensorflow_project_to_ground(image,np.zeros((int(exp_config.camera_image_patch_size[0]/(8*exp_config.downsample_ratio)),int(exp_config.camera_image_patch_size[1]/(8*exp_config.downsample_ratio)))),pose_ref, calib_ref,pose_cur,calib_cur, cam_intrinsic,reference_frame=is_reference_sample)
    warp_trans3 = utils.tensorflow_project_to_ground(image,np.zeros((int(exp_config.camera_image_patch_size[0]/(16*exp_config.downsample_ratio)),int(exp_config.camera_image_patch_size[1]/(16*exp_config.downsample_ratio)))),pose_ref, calib_ref,pose_cur,calib_cur, cam_intrinsic,reference_frame=is_reference_sample)
    
   
    
    warped_img, warped_cover, coordinate_transform = utils.argoverse_project_to_ground(exp_config, image,image_labels[...,exp_config.num_bev_classes],calib_ref,pose_ref,calib_cur,pose_cur,cam_intrinsic,reference_frame=is_reference_sample)
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

#    train_file ='C:\\winpython\\WPy-3670\\codes\\davis2017\\DAVIS\\ImageSets\\2017\\train.txt'
#    data_images_path ='C:\\winpython\\WPy-3670\\codes\\davis2017\\DAVIS\\JPEGImages\\480p\\drone'
    
    
    logging.error('EXPERIMENT : ' + str(exp_config.experiment_name))
    logging.error('THIS IS : ' + str(log_dir))

    val_tokens = token_splits.VAL_LOGS

    logging.info('EXPERIMENT NAME: %s' % exp_config.experiment_name)



    # Tell TensorFlow that the model will be built into the default Graph.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
#        with tf.Graph().as_default():
    with tf.Session(config = config) as sess:
        # Generate placeholders for the images and labels.

        
        
        training_time_placeholder = tf.placeholder(tf.bool, shape=[])
        
        
        my_training_placeholder = tf.placeholder(tf.bool, shape=[])
        
        num_frames = None
        n_frames_per_seq = None
        
        reference_frame_index_pl = tf.placeholder(tf.int32, shape=[])
        
        
        # Build a Graph that computes predictions from the inference model.
        my_model_options = common.ModelOptions({common.OUTPUT_TYPE:10},crop_size=exp_config.camera_image_patch_size,atrous_rates=[6, 12, 18])
   
        image_tensor_shape = [n_frames_per_seq,exp_config.camera_image_patch_size[0],exp_config.camera_image_patch_size[1],3]
        image_mask_tensor_shape = [n_frames_per_seq,int(exp_config.camera_image_patch_size[0]/4),int(exp_config.camera_image_patch_size[1]/4),total_label_slices]
        # mask_tensor_shape = [n_seqs,exp_config.patch_size[1],exp_config.patch_size[0],exp_config.num_bev_classes + 1]
        
        images_placeholder = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
        
        image_labels_placeholder = tf.placeholder(tf.float32, shape=image_mask_tensor_shape, name='image_labels')
        
        separate_covers_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,exp_config.patch_size[1],exp_config.patch_size[0],1], name='separate_covers')
        
        bev_transforms_placeholder = tf.placeholder(tf.float32, shape=[np.max([1,n_seqs-1]),8], name='bev_transforms')
        
        ground_transforms_placeholder1 = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,8], name='ground_transforms1')
        
        ground_transforms_placeholder2 = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,8], name='ground_transforms2')
        
        ground_transforms_placeholder3 = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,8], name='ground_transforms3')
        
        
        coordinate_ground_transforms_placeholder = tf.placeholder(tf.float32, shape=[n_seqs,num_frames,3,3], name='coordinate_ground_transforms')
        
       
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
            reference_image_endpoints.append(tf.slice(image_total_relative_endpoints[endi],[reference_frame_index_pl,0,0,0],[1,-1,-1,-1]))
        
        side_obj_logits, side_obj_softmaxed = mem_net.my_object_side_decoder(reference_image_endpoints,tf.slice(total_input_image,[reference_frame_index_pl,0,0,0],[1,-1,-1,-1]),exp_config,reuse=False)
        # logging.error('SIDE OCC LOGITS ' + str(side_obj_))
        # logging.error('SIDE OCC LABELS ' + str(tf.squeeze(tf.slice(image_labels_placeholder,[0,0,0,exp_config.num_classes+1],[-1,-1,-1,-1]),axis=-1)))
        
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
            tf.squeeze(tf.slice(ground_transforms_placeholder1,[0,reference_frame_index_pl,0],[1,1,-1]),axis=0),
            interpolation='BILINEAR',
            output_shape=(exp_config.project_patch_size[1],exp_config.project_patch_size[0]),
            name='tensorflow_ground_transform'
        )
        projected_obj_estimates = projected_obj_estimates*tf.squeeze(tf.slice(separate_covers_placeholder,[0,reference_frame_index_pl,0,0,0],[1,1,-1,-1,-1]),axis=0)
        
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
        
        combined_back_out = tf.concat([combined_back_out,tf.slice(all_bev_total_backbone_out,[reference_frame_index_pl,0,0,0],[1,-1,-1,-1])],axis=-1)
        
        
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
        
        combined_end = tf.concat([combined_end,tf.slice(all_bev_end2,[reference_frame_index_pl,0,0,0],[1,-1,-1,-1])],axis=-1)
        
        # combined_end = tf.reduce_max( all_bev_end2*cur_separate_covers,axis=0,keepdims=True)
        
        combined_end = tf.image.resize(
                combined_end, [int(exp_config.patch_size[1]/4),int(exp_config.patch_size[0]/4)] ,method='bilinear',name='projected_estimates_resize'  )
        
        bev_total_relative_endpoints = [tf.concat([combined_end,bigger_resized_combined_projected_estimates],axis=-1)]
        
       
        
        total_input = tf.concat([ resized_combined_projected_estimates,bev_total_backbone_out],axis=-1)
        
        
        
        static_logits, static_masks,object_logits, object_masks = mem_net.my_bev_object_decoder(bev_total_relative_endpoints,total_input,exp_config,reuse=False)
        
        masks = tf.concat([static_masks,object_masks],axis=-1)
        saver = tf.train.Saver(max_to_keep=2)
        
        # saver_best_loss = tf.train.Saver(max_to_keep=2)
        init = tf.global_variables_initializer()
        sess.run(init)
        
        
        load_path = exp_config.load_path

        saver.restore(sess,load_path)
            # to_load_saver.restore(sess,load_path)
        sess.run(mem_net.interp_surgery(tf.global_variables()))
        
        
      
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
     reference_frame_index_pl,
      combined_projected_estimates,
     0,training_time_placeholder,val_folder_path=validation_res_path)
        overall_mean = np.mean(np.array(val_res))
        logging.error('Overall mean : ' + str(overall_mean))

 

def eval_iterator(ref_id,frame_interval,n_frames_per_seq,my_scene,cur_index,single_frame=False,apply_interval=False): 
    
    num_frames = n_frames_per_seq
    
    n_seqs = 1
    
    
    camera = "ring_front_center"
    scene = train_loader.get(my_scene)
    
    pool = ThreadPool(n_seqs*num_frames) 
    
    frame_ids=[]
    first_frame = cur_index

    frame_ids.append(first_frame)
    
    # logging.error('LEN IMAGES ' + str(len(all_images_list)))
    
    
    if single_frame:
        
        for frame_number in range(1,n_frames_per_seq):
            frame_ids.append(first_frame )
#            logging.error('ENTERED SINGLE FRAME ' + str(frame_ids))
            
    else:
        # if apply_interval:
        for frame_number in range(1,n_frames_per_seq):
            frame_ids.append(first_frame + frame_interval*frame_number)
        
        
        # else:    
        #     for frame_number in range(1,n_frames_per_seq):
        #         frame_ids.append(first_frame + frame_number)
        
    pairs = []
    
    # logging.error('SCENE ' + my_scene)
    # logging.error('FRAMES ' + str(frame_ids))

    
   
    timestamp = str(np.copy(train_loader._image_timestamp_list_sync[my_scene][camera][frame_ids[ref_id]]))

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
    
    
    pose_ref = np.copy(scene.get_pose(frame_ids[ref_id]).transform_matrix)
    for k in range(n_seqs):
        for m in range(num_frames):
            timestamp = str(np.copy(train_loader._image_timestamp_list_sync[my_scene][camera][frame_ids[m]]))
            # logging.error('TIME ' + str(m) + ' ' + timestamp)
            pose_cur = np.copy(scene.get_pose(frame_ids[m]).transform_matrix)
            output_path = os.path.join(exp_config.argo_labels_path, 
                                       my_scene, camera, 
                                       str(camera)+'_'+str(timestamp)+'.png')
            
            image_string = os.path.join(exp_config.argo_track_path,my_scene,'ring_front_center','ring_front_center_'+str(timestamp)+'.jpg')

            
            pairs.append((image_string,output_path,vis_mask,calib_cur,pose_ref,pose_cur,m==ref_id))
    
   

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

def overall_eval_iterator(ref_id,frame_interval,n_frames_per_seq,my_scene,cur_index, single_frame=False,apply_interval=False):

    seq_images_ar, seq_labels_ar,bev_images_ar,bev_covers_ar,_, tf_transforms1,tf_transforms2,tf_transforms3,coordinate_transforms_ar,real_ref_bev_labels,to_image_transform,all_bev_labels_ar,went_well =eval_iterator(ref_id,frame_interval,n_frames_per_seq,my_scene,cur_index,single_frame=single_frame,apply_interval=apply_interval)
     

    return seq_images_ar,  tf_transforms1,tf_transforms2,tf_transforms3, bev_covers_ar, coordinate_transforms_ar,real_ref_bev_labels



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
                       reference_frame_index_pl,
                        combined_projected_estimates,
                        iteration,training_time_placeholder,val_folder_path=validation_res_path):


    logging.error('Started evaluation')
    
    
    res_strings=[]

    all_static_j1s=[]
    all_object_j1s=[]

    for my_scene_token in val_tokens:
        
        scene_static_results=[]
        scene_object_results=[]

        name_of_seq = my_scene_token
        

        scene = train_loader.get(my_scene_token)
        # logging.error('SCENE ' + my_scene_token)
        n_frames_in_scene = scene.num_lidar_frame
        if not os.path.exists(os.path.join(validation_res_path,name_of_seq)):
            os.makedirs(os.path.join(validation_res_path,name_of_seq))
        logging.error('N FRAMES  ' + str(n_frames_in_scene))
        
        
        for frame_number in range(n_frames_in_scene):
#            logging.error('FRAME NUMBER ' + str(frame_number))
            if single_frame_experiment:
                batch_image, batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_ref_bev_labels =\
                overall_eval_iterator(my_scene_token,frame_number,single_frame=True,apply_interval=False)
            else:
                if frame_number < frame_interval*reference_frame_index: 

                    ref_ind = 0
                    batch_image, batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_ref_bev_labels =\
                    overall_eval_iterator(ref_ind, frame_interval,num_frames, my_scene_token,frame_number,single_frame=False,apply_interval=True)
                
                elif (frame_number == (n_frames_in_scene - 1)):
#
                    ref_ind = 0
                    batch_image, batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_ref_bev_labels =\
                    overall_eval_iterator(ref_ind,  1 ,3, my_scene_token,frame_number,single_frame=True,apply_interval=False)
                
                elif (frame_number >= (n_frames_in_scene - (num_frames - reference_frame_index - 1))):
                    

                    ref_ind = 0
                    n_frames = n_frames_in_scene - frame_number 
                    batch_image, batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_ref_bev_labels =\
                    overall_eval_iterator(ref_ind,  1 ,n_frames, my_scene_token,frame_number,single_frame=False,apply_interval=False)
                    
                elif (frame_number >= (n_frames_in_scene - frame_interval*(num_frames - reference_frame_index - 1))):
                    
                    
                    for te in range(num_frames - reference_frame_index - 1,-1,-1):
                        if frame_number + frame_interval*te < (n_frames_in_scene):
                            break
                    
                    
                    
                    if te == 0:
                        ref_ind = 0
                        batch_image, batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_ref_bev_labels =\
                        overall_eval_iterator(ref_ind,  1 ,2, my_scene_token,frame_number,single_frame=True,apply_interval=False)
                    
                    else:
                        ref_ind = reference_frame_index
                        batch_image, batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_ref_bev_labels =\
                        overall_eval_iterator(ref_ind,  frame_interval,reference_frame_index + te, my_scene_token, int(frame_number - frame_interval*reference_frame_index),single_frame=False,apply_interval=False)
                        
                else:
                    
                    # logging.error('MULTI FRAME YES INTERVAL ' + str(int(frame_number - frame_interval*reference_frame_index)))
                    
                    ref_ind = reference_frame_index
                    batch_image, batch_tf_transforms1,batch_tf_transforms2,batch_tf_transforms3, batch_separate_covers ,batch_coordinate_transforms ,batch_ref_bev_labels =\
                    overall_eval_iterator(ref_ind,  frame_interval,num_frames, my_scene_token,int(frame_number - frame_interval*reference_frame_index),single_frame=False,apply_interval=True)
                    
            # logging.error('WE GOT THE STUFF')
            feed_dict = {
                                
           
            training_time_placeholder: False,
            my_training_placeholder:False,
            
            images_placeholder:batch_image,
              
            bev_transforms_placeholder:np.zeros((1,8)),
            reference_frame_index_pl: np.int32(ref_ind),
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

            thresh_list=[0.5, 0.55, 0.45, 0.7, 0.5, 0.1, 0.25, 0.5]
            
            sample_results=[]
            squeezed_masks = np.squeeze(masks_v)
            static_estimates = np.uint8(squeezed_masks[...,:exp_config.num_static_classes] > thresh_list[0])
            
            temp_object_estimates = squeezed_masks[...,exp_config.num_static_classes:exp_config.num_bev_classes]
            
            # logging.error('TEMP OBJ ' + str(temp_object_estimates))
            
            object_estimates = np.zeros((temp_object_estimates.shape[0],temp_object_estimates.shape[1],exp_config.num_object_classes))
            for k in range(exp_config.num_object_classes):
                object_estimates[...,k] = np.uint8(temp_object_estimates[...,k] > (thresh_list[k+1]))
                
            bg_estimate = 1 - np.clip(np.sum(object_estimates,axis=-1,keepdims=True),0,1)
            
            hard_estimates = np.concatenate([static_estimates,object_estimates],axis=-1)
            
            object_estimates = np.concatenate([object_estimates,bg_estimate],axis=-1)
            
            object_stats = utils.get_confusion(np.squeeze(batch_ref_bev_labels[...,exp_config.num_static_classes:exp_config.num_bev_classes]), object_estimates,np.squeeze(batch_ref_bev_labels[...,exp_config.num_bev_classes+1]),mask_iou=exp_config.use_occlusion)
             
            
            
            for k in range(exp_config.num_bev_classes):
      
                all_stats , void_pixels= utils.get_all_stats(np.squeeze(batch_ref_bev_labels[...,k]), hard_estimates[...,k],np.squeeze(batch_ref_bev_labels[...,exp_config.num_bev_classes+1]),mask_iou=exp_config.use_occlusion)
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
        # temp_string = " : Scene " + str(my_scene_token)+' best res : '+str(np.max(np.sum(temp_res,axis=-1)/np.sum(exists,axis=-1),axis=0)) + ' at ' +str(frame_numbers_list[np.argmax(np.sum(temp_res,axis=-1)/np.sum(exists,axis=-1),axis=0)]) 
        
        
        
        res_strings.append(temp_string)
        logging.error(temp_string)
        write_to_txt_file(os.path.join(log_dir,'val_results.txt'),[temp_string])
    
    # to_return = all_j1s
    
    # logging.error('ALL J1 ' + str(all_j1s))
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
    
    
    temp_string = 'Static j : ' + str(confuse_iou) + '\n' +\
    ' Static tp_rate : ' + str(tp_rate)+ '\n' +' Static fp_rate : ' + str(fp_rate)+ '\n'+ ' Static precision : ' + str(precision)+ '\n'+'Object confuse : ' + str(object_raw)
    
    logging.error(temp_string)
    res_strings.append(temp_string)
    write_to_txt_file(os.path.join(log_dir,'confuse_results.txt'),res_strings)
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
