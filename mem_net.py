from __future__ import print_function
"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import tensorflow as tf
import numpy as np

from PIL import Image

import layers
import logging
from deeplab import model

slim = tf.contrib.slim
            


def full_object_loss(y_pred,labels,occ_mask,exp_config,alpha_pos,weight=True,weight_vector=None, focal=True):
    
   
    pen_mask = occ_mask
    alpha_neg = tf.constant(np.ones_like(alpha_pos), tf.float32)
    
    
    
    gamma = 2
    epsilon = 0.00001
    
    # y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)

    we = 0.5
    L=alpha_pos*(-labels*we*(tf.pow((1-y_pred),gamma))*tf.log(y_pred + epsilon)-\
      (1-labels)*(1-we)*(tf.pow(y_pred,gamma))*tf.log(1-y_pred + epsilon))
    
    # L = L*pen_mask
    return pen_mask*L, alpha_pos 


def classwise_object_loss(y_pred,labels,occ_mask,exp_config,alpha_pos,weight=True,weight_vector=None, focal=True):
    
   
    # pen_mask = occ_mask + (1-occ_mask)*0.1
    pen_mask = occ_mask
    alpha_neg = tf.constant(np.ones_like(alpha_pos), tf.float32)
    
    
    
    gamma = 2
    epsilon = 0.00001
    
    # y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
#   
    we = 0.5
    L=-alpha_pos*labels*we*(tf.pow((1-y_pred),gamma))*tf.log(y_pred + epsilon)-\
      (1-labels)*alpha_neg*(1-we)*(tf.pow(y_pred,gamma))*tf.log(1-y_pred + epsilon)
    
    # L = L*pen_mask
    return pen_mask*L, alpha_pos 



def contrastive_sigmoid_loss(logits,labels,exp_config,weight=True,weight_vector=None, focal=True):
    
    labels_shape = labels.get_shape().as_list()

    vis_mask = tf.slice(labels,[0,0,0,exp_config.num_classes],[-1,-1,-1,1])
    occ_mask = tf.slice(labels,[0,0,0,exp_config.num_classes+1],[-1,-1,-1,1])
    
    tot_mask = 1-occ_mask*vis_mask
    
#    if exp_config.use_occlusion:
#    
#        tot_mask = occ_mask*vis_mask
#    else:
#        tot_mask = vis_mask
    
    labels=tf.slice(labels,[0,0,0,0],[-1,-1,-1,exp_config.num_classes])
    
    # pen_mask = tf.ones_like(labels)
    # temp_mask = tot_mask*labels
    pen_mask = tot_mask*0.1 + (1-tot_mask)
    pen_mask = tf.to_float(pen_mask)
#    pen_mask = occ_mask*labels*0.1 + (1-occ_mask)*labels
#    no_labels = 1 - tf.clip_by_value(tf.reduce_sum(labels,axis=-1,keepdims=True),0,1)
    
    alpha_pos = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(np.array([0.5,2.5,2.0,3.0,0.5,0.5]),axis=0),axis=0),axis=0), tf.float32)
    alpha_neg = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(np.array([0.5,1.0,1.0,1.0,0.5,0.5]),axis=0),axis=0),axis=0), tf.float32)
    
    
    
    gamma = 2
    epsilon = 0.00001
    
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
#    no_labels=tf.to_float(no_labels)
    
#    no_labels_loss = 
    
    
#    L = tf.nn.sigmoid_cross_entropy_with_logits(
#        labels=labels, logits=logits, name=None
#    )
    
    
    L=-labels*alpha_pos*(tf.pow((1-y_pred),gamma))*tf.log(y_pred + epsilon)-\
      (1-labels)*alpha_neg*(tf.pow(y_pred,gamma))*tf.log(1-y_pred + epsilon)
    
    L = L*pen_mask
    return L, alpha_pos


def argoverse_contrastive_sigmoid_loss(y_pred,labels,exp_config,weight=True,weight_vector=None, focal=True):
    
    labels_shape = labels.get_shape().as_list()

    vis_mask = tf.slice(labels,[0,0,0,exp_config.num_classes],[-1,-1,-1,1])
    occ_mask = tf.slice(labels,[0,0,0,exp_config.num_classes+1],[-1,-1,-1,1])
    
    tot_mask = occ_mask
    
#    if exp_config.use_occlusion:
#    
#        tot_mask = occ_mask*vis_mask
#    else:
#        tot_mask = vis_mask
    
    labels=tf.slice(labels,[0,0,0,0],[-1,-1,-1,1])
    
    # pen_mask = tf.ones_like(labels)
    # temp_mask = tot_mask*labels
    # pen_mask = tot_mask*0.1 + (1-tot_mask)
    pen_mask = tf.to_float(tot_mask)
#    pen_mask = occ_mask*labels*0.1 + (1-occ_mask)*labels
#    no_labels = 1 - tf.clip_by_value(tf.reduce_sum(labels,axis=-1,keepdims=True),0,1)
    
    # alpha_pos = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(np.array([0.5,2.5,2.0,3.0,0.5,0.5]),axis=0),axis=0),axis=0), tf.float32)
    # alpha_neg = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(np.array([0.5,1.0,1.0,1.0,0.5,0.5]),axis=0),axis=0),axis=0), tf.float32)
    
    
    alpha_pos=0.7
    alpha_neg=0.5
    gamma = 2
    epsilon = 0.00001
    
   
    
    L=-labels*alpha_pos*(tf.pow((1-y_pred),gamma))*tf.log(y_pred + epsilon)-\
      (1-labels)*alpha_neg*(tf.pow(y_pred,gamma))*tf.log(1-y_pred + epsilon)
    
    L = L*pen_mask
    return L, alpha_pos


def occlusion_loss(logits, labels,depth=2):
    labels = tf.one_hot(tf.cast(tf.squeeze(labels),tf.uint8),depth=depth)
#    temp = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
#    loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(temp,axis=[1,2]),axis=[1]))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    return loss




def full_modified_bev_object_loss(y_pred,labels,cover,exp_config,alpha_pos,alpha_neg,img_shape=[448,800],weight=True,weight_vector=None, focal=True):
    
    
    # tot_mask = 1-tf.slice(labels,[0,0,0,exp_config.num_bev_classes],[-1,-1,-1,1])
    tot_mask = tf.slice(labels,[0,0,0,exp_config.num_bev_classes+1],[-1,-1,-1,1])
    
    # alpha_pos = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(exp_config.bev_positive_weights,axis=0),axis=0),axis=0), tf.float32)
    # alpha_neg = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(exp_config.bev_negative_weights,axis=0),axis=0),axis=0), tf.float32)
    gamma = 2
    epsilon = 0.00001
#    
    
    labels=tf.slice(labels,[0,0,0,0],[-1,-1,-1,exp_config.num_bev_classes])

    object_labels = tf.slice(labels,[0,0,0,exp_config.num_static_classes],[-1,-1,-1,-1])

    
    bg_labels = 1-tf.clip_by_value(tf.reduce_sum(object_labels,axis=-1,keepdims=True),0,1)
    # bg_labels = tf.zeros_like(tf.slice(object_labels,[0,0,0,0],[-1,-1,-1,1]))
    labels=tf.concat([labels,bg_labels],axis=-1)
    labels=tf.to_float(labels)
    
    we = 0.6
    L=alpha_pos*(-labels*we*(tf.pow((1-y_pred),gamma))*tf.log(y_pred + epsilon)-\
      (1-labels)*(1-we)*(tf.pow(y_pred,gamma))*tf.log(1-y_pred + epsilon))
    
    # L=-labels*alpha_pos*(tf.pow((1-y_pred),gamma))*tf.log(y_pred + epsilon)-\
    #   (1-labels)*alpha_neg*(tf.pow(y_pred,gamma))*tf.log(1-y_pred + epsilon)
      
    L = L*tot_mask
        
    return L, alpha_pos


def classwise_modified_bev_object_loss(y_pred,labels,cover,exp_config,alpha_pos,alpha_neg,img_shape=[448,800],weight=True,weight_vector=None, focal=True):
    
    
    # tot_mask = 1-tf.slice(labels,[0,0,0,exp_config.num_bev_classes],[-1,-1,-1,1])
    tot_mask = tf.slice(labels,[0,0,0,exp_config.num_bev_classes+1],[-1,-1,-1,1])
    
    # alpha_pos = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(exp_config.bev_positive_weights,axis=0),axis=0),axis=0), tf.float32)
    # alpha_neg = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(exp_config.bev_negative_weights,axis=0),axis=0),axis=0), tf.float32)
    gamma = 2
    epsilon = 0.00001
#    
    
    labels=tf.slice(labels,[0,0,0,0],[-1,-1,-1,exp_config.num_bev_classes])

    object_labels = tf.slice(labels,[0,0,0,exp_config.num_static_classes],[-1,-1,-1,-1])
    
 
#     labels=tf.to_float(labels)
# #    alpha_pos = alpha_pos[:-1]
#     y_pred=tf.slice(y_pred,[0,0,0,0],[-1,-1,-1,exp_config.num_bev_classes])
    
#     we = 0.6
#     L=alpha_pos*(-labels*we*(tf.pow((1-y_pred),gamma))*tf.log(y_pred + epsilon)-\
#       (1-labels)*(1-we)*(tf.pow(y_pred,gamma))*tf.log(1-y_pred + epsilon))
  
      
#     L = L*tot_mask
    
    
    bg_labels = 1-tf.clip_by_value(tf.reduce_sum(object_labels,axis=-1,keepdims=True),0,1)
    # bg_labels = tf.zeros_like(tf.slice(object_labels,[0,0,0,0],[-1,-1,-1,1]))
    labels=tf.concat([labels,bg_labels],axis=-1)
    labels=tf.to_float(labels)
    
    # we = 0.6
    # L=alpha_pos*(-labels*we*(tf.pow((1-y_pred),gamma))*tf.log(y_pred + epsilon)-\
    #   (1-labels)*(1-we)*(tf.pow(y_pred,gamma))*tf.log(1-y_pred + epsilon))
    
    L=-labels*alpha_pos*(tf.pow((1-y_pred),gamma))*tf.log(y_pred + epsilon)-\
      (1-labels)*alpha_neg*(tf.pow(y_pred,gamma))*tf.log(1-y_pred + epsilon)
      
    L = L*tot_mask
        
    return L, alpha_pos


def linear_activation(x):
    '''
    A linear activation function (i.e. no non-linearity)
    '''
    return x

def crop_features(feature, out_size):
    """Crop the center of a feature map
    Args:
    feature: Feature map to crop
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    """
    up_size = tf.shape(feature)
    ini_w = tf.div(tf.subtract(up_size[1], out_size[1]), 2)
    ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2)
    slice_input = tf.slice(feature, (0, ini_w, ini_h, 0), (-1, out_size[1], out_size[2], -1))
    # slice_input = tf.slice(feature, (0, ini_w, ini_w, 0), (-1, out_size[1], out_size[2], -1))  # Caffe cropping way
    return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])

def pixel_wise_cross_entropy_loss(logits, labels,depth=2):
    '''
    Simple wrapper for the normal tensorflow cross entropy loss 
    '''
    labels = tf.one_hot(labels,depth=depth)
#    temp = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
#    loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(temp,axis=[1,2]),axis=[1]))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    return loss


def bev_occlusion_loss(logits, labels,mask,depth=2):
    labels = tf.one_hot(tf.cast(tf.squeeze(labels),tf.uint8),depth=depth)
#    temp = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
#    loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(temp,axis=[1,2]),axis=[1]))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)*mask)
    return loss
    
def masked_pixel_wise_cross_entropy_loss(logits, labels,mask,depth=2):
    '''
    Simple wrapper for the normal tensorflow cross entropy loss 
    '''
    labels = tf.one_hot(tf.cast(labels,tf.uint8),depth=depth)
#    temp = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
#    loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(temp,axis=[1,2]),axis=[1]))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)*mask)
    return loss

def pixel_wise_cross_entropy_loss_weighted(logits, labels, class_weights):
    '''
    Weighted cross entropy loss, with a weight per class
    :param logits: Network output before softmax
    :param labels: Ground truth masks
    :param class_weights: A list of the weights for each class
    :return: weighted cross entropy loss
    '''


    flat_logits = tf.reshape(logits, [-1, 2])
    flat_labels = tf.reshape(labels, [-1, 2])

    class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

    weight_map = tf.multiply(flat_labels, class_weights)
    weight_map = tf.reduce_sum(weight_map, axis=1)

    loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
    weighted_loss = tf.multiply(loss_map, weight_map)

    loss = tf.reduce_mean(weighted_loss)

    return loss

def sigmoid_loss(logits,labels,exp_config,img_shape=[448,800],weight=True,weight_vector=None, focal=True):
    
    
    vis_mask = tf.slice(labels,[0,0,0,exp_config.num_classes],[-1,-1,-1,1])
    occ_mask = tf.slice(labels,[0,0,0,exp_config.num_classes+1],[-1,-1,-1,1])
    
    if exp_config.use_occlusion:
    
        tot_mask = occ_mask*vis_mask
    else:
        tot_mask = vis_mask
    labels=tf.slice(labels,[0,0,0,0],[-1,-1,-1,exp_config.num_classes])
    

    
    alpha = tf.constant(np.expand_dims(np.expand_dims(np.expand_dims(np.array([0.5,0.6,0.7,0.5,0.5,0.5]),axis=0),axis=0),axis=0), tf.float32)
    gamma = 2
    epsilon = 0.00001
    
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-labels*alpha*(tf.pow((1-y_pred),gamma))*tf.log(y_pred + epsilon)-\
      (1-labels)*(1-alpha)*(tf.pow(y_pred,gamma))*tf.log(1-y_pred + epsilon)
    
    L = L*tot_mask
    
    return L, alpha
    



def soft_focal_loss(sigmoids,labels):
    
    alpha = 0.9
    gamma = 2
    epsilon = 0.00001
    beta = 4
    
    hard_labels = tf.cast(tf.greater_equal(labels,0.7),tf.float32)
    y_pred = sigmoids
#    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-hard_labels*alpha*tf.pow((1-y_pred),gamma)*tf.log(y_pred + epsilon)-\
      (1-hard_labels)*(1-alpha)*tf.pow(1-labels,beta)*tf.pow(y_pred,gamma)*tf.log(1-y_pred + epsilon)

    return L


def deeplab_residual_block_RGMP(input_,num_filters,reuse=False):
 
    
    relu_temp = tf.nn.relu(input_)
    res1 = layers.coord_conv2D_layer(bottom=relu_temp,
                    name='res1',
                    
                    num_filters=num_filters,
                    
                    activation=linear_activation)
    relu_res1 = tf.nn.relu(res1)
    res2 = layers.coord_conv2D_layer(bottom=relu_res1,
                    name='res2',
                    
                    num_filters=num_filters,
                    
                    activation=linear_activation)
    out = input_ + res2
    return out
def deeplab_refinement_module(input_, skip,num_filters,reuse=False):
    with tf.variable_scope('refinement_upscale', reuse=reuse):
        upsampled = tf.contrib.layers.conv2d_transpose(
            input_,
            num_filters,
            4,
            stride=2,
            padding='SAME',
            activation_fn=None,
            trainable=False   
        )
    temp = layers.coord_conv2D_layer(bottom=skip,
                name='temp_conv',
                
                num_filters=num_filters,
                
                activation=linear_activation)
    with tf.variable_scope('residual_block1', reuse=reuse):
        res_out1 = residual_block_RGMP(temp,num_filters)
        
    temp_sum = res_out1 + upsampled
    with tf.variable_scope('residual_block2', reuse=reuse):
        out = residual_block_RGMP(temp_sum,num_filters)
    
    
    return out

def residual_block_RGMP(input_,num_filters,reuse=False):
 
    
    relu_temp = tf.nn.relu(input_)
    res1 = layers.conv2D_layer(bottom=relu_temp,
                    name='res1',
                    
                    num_filters=num_filters,
                    
                    activation=linear_activation)
    relu_res1 = tf.nn.relu(res1)
    res2 = layers.conv2D_layer(bottom=relu_res1,
                    name='res2',
                    
                    num_filters=num_filters,
                    
                    activation=linear_activation)
    out = input_ + res2
    return out
def refinement_module(input_, skip,num_filters,reuse=False):
    with tf.variable_scope('refinement_upscale', reuse=reuse):
        upsampled = tf.contrib.layers.conv2d_transpose(
            input_,
            num_filters,
            4,
            stride=2,
            padding='SAME',
            activation_fn=None,
            trainable=False   
        )
    temp = layers.conv2D_layer(bottom=skip,
                name='temp_conv',
                
                num_filters=num_filters,
                
                activation=linear_activation)
    with tf.variable_scope('residual_block1', reuse=reuse):
        res_out1 = deeplab_residual_block_RGMP(temp,num_filters)
        
    temp_sum = res_out1 + upsampled
    with tf.variable_scope('residual_block2', reuse=reuse):
        out = deeplab_residual_block_RGMP(temp_sum,num_filters)
    
    
    return out

def get_side_prediction(post_processed,reuse=False,scope='side_estimator'):
    num_filters=128
    y=post_processed
    with tf.variable_scope(scope, reuse=reuse):
        y = layers.conv2D_layer(bottom=y,
                name='init_conv',
                
                num_filters=num_filters,
                
                activation=tf.nn.relu)
        with tf.variable_scope('init_res_block', reuse=reuse):
            y=residual_block_RGMP(y, num_filters)
        
        for k in range(2):
            with tf.variable_scope('module'+str(k), reuse=reuse):
                y=residual_block_RGMP(y, num_filters)
                y = tf.contrib.layers.conv2d_transpose(
                    y,
                    num_filters,
                    4,
                    stride=2,
                    padding='SAME',
                    activation_fn=None,
                    trainable=False   
                )
   
        
        temp = layers.coord_conv2D_layer(bottom=y,
                name='last_conv1',
                
                num_filters=6,
                
                activation=tf.identity)
     
    
    return temp, tf.nn.sigmoid(temp)

def vertical_residual_block(input_,num_filters,reuse=False):
 
    
    relu_temp = tf.nn.relu(input_)
    res1 = layers.coord_conv2D_layer(bottom=relu_temp,
                    name='vert1',
                    kernel_size=(7,1),
                    num_filters=num_filters,
                    
                    activation=linear_activation)
    relu_res1 = tf.nn.relu(res1)
    res2 = layers.coord_conv2D_layer(bottom=relu_res1,
                    name='vert2',
                    kernel_size=(7,1),
                    num_filters=num_filters,
                    
                    activation=linear_activation)
    out = input_ + res2
    return out

def my_bev_object_decoder(endpoints, y,exp_config,apply_softmax=False, reuse=False, scope='my_bev_object_decoder'):
    """Produces the mask from the encoder endpoints and the memory read output
    Args:
    endpoints: The output of the residual block in each stage of the encoder ()
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    
    """
    
    num_filters = 128
    
#    layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
    with tf.variable_scope(scope, reuse=reuse):
        y = layers.coord_conv2D_layer(bottom=y,
                name='init_conv',
                
                num_filters=num_filters,
                
                activation=linear_activation)
        
        
        
        with tf.variable_scope('init_res_block', reuse=reuse):
            y_base=deeplab_residual_block_RGMP(y, num_filters)
            
        with tf.variable_scope('init_vert_block', reuse=reuse):
            y_base = vertical_residual_block(y_base,num_filters)
        
        with tf.variable_scope('init_res_block2', reuse=reuse):
            y_base=deeplab_residual_block_RGMP(y_base, num_filters)
        
        with tf.variable_scope('init_vert_block2', reuse=reuse):
            y_base = vertical_residual_block(y_base,num_filters)
        
        # with tf.variable_scope('init_res_block3', reuse=reuse):
        #     y_base=deeplab_residual_block_RGMP(y_base, num_filters)
            
            
        processed_endpoint = endpoints[0]
        processed_endpoint = layers.coord_conv2D_layer(bottom=processed_endpoint,
            name='processed_endpoint_init_conv',
            
            num_filters=num_filters,
            
            activation=linear_activation)
        with tf.variable_scope('processed_endpoint_init_res_block1', reuse=reuse):
            processed_endpoint=deeplab_residual_block_RGMP(processed_endpoint, num_filters)
        
        with tf.variable_scope('processed_vert_block', reuse=reuse):
            y_base = vertical_residual_block(y_base,num_filters)
        with tf.variable_scope('processed_endpoint_init_res_block2', reuse=reuse):
            processed_endpoint=deeplab_residual_block_RGMP(processed_endpoint, num_filters)
        
        with tf.variable_scope('processed_vert_block2', reuse=reuse):
            y_base = vertical_residual_block(y_base,num_filters)
        # with tf.variable_scope('processed_endpoint_init_res_block3', reuse=reuse):
        #     processed_endpoint=deeplab_residual_block_RGMP(processed_endpoint, num_filters)
            
        
        # for i in range(1):
        with tf.variable_scope('refine_module1', reuse=reuse):
            y_base = refinement_module(y_base, processed_endpoint,num_filters)
        with tf.variable_scope('after_refine_block', reuse=reuse):
            y_base=deeplab_residual_block_RGMP(y_base, num_filters)
        y_base = tf.slice(y_base,[0,int(exp_config.extra_space[0]/8),int(exp_config.extra_space[1]/8),0],[-1,int(exp_config.label_patch_size[1]/4),int(exp_config.label_patch_size[0]/4),-1])
        
        
        y=y_base

   
        with tf.variable_scope('post_upscale', reuse=reuse):
            y = tf.contrib.layers.conv2d_transpose(
                y,
                num_filters,
                4,
                stride=2,
                padding='SAME',
                activation_fn=None,
                trainable=False   
            )
        
        with tf.variable_scope('class_second_res_block2', reuse=reuse):
            y=deeplab_residual_block_RGMP(y, num_filters)
          
        
        with tf.variable_scope('post_upscale2', reuse=reuse):
            y = tf.contrib.layers.conv2d_transpose(
                y,
                num_filters,
                4,
                stride=2,
                padding='SAME',
                activation_fn=None,
                trainable=False   
            )
        temp = layers.conv2D_layer(bottom=y,
                name='last_conv1',
                
                num_filters=num_filters,
                
                activation=tf.nn.relu)
        static_temp = layers.conv2D_layer(bottom=temp,
                name='last_conv_temp',
                
                num_filters=exp_config.num_static_classes,
                
                activation=linear_activation)
        
        y=y_base
        with tf.variable_scope('object_post_upscale', reuse=reuse):
            y = tf.contrib.layers.conv2d_transpose(
                y,
                num_filters,
                4,
                stride=2,
                padding='SAME',
                activation_fn=None,
                trainable=False   
            )
        
        with tf.variable_scope('object_class_second_res_block2', reuse=reuse):
            y=deeplab_residual_block_RGMP(y, num_filters)
          
        
        with tf.variable_scope('object_post_upscale2', reuse=reuse):
            y = tf.contrib.layers.conv2d_transpose(
                y,
                num_filters,
                4,
                stride=2,
                padding='SAME',
                activation_fn=None,
                trainable=False   
            )
        temp = layers.conv2D_layer(bottom=y,
                name='object_last_conv1',
                
                num_filters=num_filters,
                
                activation=tf.nn.relu)
        object_temp = layers.conv2D_layer(bottom=temp,
                name='object_last_conv_temp',
                
                num_filters=exp_config.num_object_classes+1,
                
                activation=linear_activation)

    if apply_softmax:
        return static_temp, tf.nn.sigmoid(static_temp),object_temp, tf.nn.softmax(object_temp)
    else:
        return static_temp, tf.nn.sigmoid(static_temp),object_temp, tf.nn.sigmoid(object_temp)



def my_object_side_decoder(endpoints,y,exp_config,reuse=False, apply_softmax=False, scope='object_side_decoder'):
    """Produces the mask from the encoder endpoints and the memory read output
    Args:
    endpoints: The output of the residual block in each stage of the encoder ()
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    
    """
    
    num_filters = 128
    
#    layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
    with tf.variable_scope(scope, reuse=reuse):
        y = layers.coord_conv2D_layer(bottom=y,
                name='init_conv',
                
                num_filters=num_filters,
                
                activation=linear_activation)
        with tf.variable_scope('init_res_block', reuse=reuse):
            y=deeplab_residual_block_RGMP(y, num_filters)
        
 
        for i in range(2):
            with tf.variable_scope('refine_module'+str(i), reuse=reuse):
                y = refinement_module(y, endpoints[i],num_filters)
            
       
        with tf.variable_scope('object_part', reuse=reuse):
            y=deeplab_residual_block_RGMP(y, num_filters)
        
        
   
        
        temp_obj = layers.conv2D_layer(bottom=y,
                name='obj_last_conv1',
                
                num_filters=num_filters,
                
                activation=tf.nn.relu)
        temp_obj = layers.conv2D_layer(bottom=temp_obj,
                name='obj_last_conv_temp',
                
                num_filters=exp_config.num_object_classes+1,
                
                activation=linear_activation)
    

    if apply_softmax:
        
        return temp_obj, tf.nn.softmax(temp_obj)
    else:
        return temp_obj, tf.nn.sigmoid(temp_obj)



def compat_my_side_decoder(endpoints,y,reuse=False,num_classes=6, scope='side_decoder'):
    """Produces the mask from the encoder endpoints and the memory read output
    Args:
    endpoints: The output of the residual block in each stage of the encoder ()
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    
    """
    
    num_filters = 128
    
#    layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
    with tf.variable_scope(scope, reuse=reuse):
        y = layers.coord_conv2D_layer(bottom=y,
                name='init_conv',
                
                num_filters=num_filters,
                
                activation=linear_activation)
        with tf.variable_scope('init_res_block', reuse=reuse):
            y=deeplab_residual_block_RGMP(y, num_filters)
        
 
        for i in range(2):
            with tf.variable_scope('refine_module'+str(i), reuse=reuse):
                y = refinement_module(y, endpoints[i],num_filters)
            
        y_base = y
        
        with tf.variable_scope('occlusion_part', reuse=reuse):
            y=deeplab_residual_block_RGMP(y, num_filters)
        
   
        
        temp = layers.conv2D_layer(bottom=y,
                name='occlusion_last_conv1',
                
                num_filters=num_filters,
                
                activation=tf.nn.relu)
        temp_occlusion = layers.conv2D_layer(bottom=temp,
                name='occlusion_last_conv_temp',
                
                num_filters=2,
                
                activation=linear_activation)


        y=y_base
        
        with tf.variable_scope('mask_part', reuse=reuse):
            y=deeplab_residual_block_RGMP(y, num_filters)
        
        
   
        
        temp = layers.conv2D_layer(bottom=y,
                name='last_conv1',
                
                num_filters=num_filters,
                
                activation=tf.nn.relu)
        
        if num_classes == 1:
            temp = layers.conv2D_layer(bottom=temp,
                name='last_conv_temp',
                
                num_filters=2,
                
                activation=linear_activation)
            
            # return temp, temp_occlusion, tf.nn.sigmoid(temp), tf.nn.softmax(temp_occlusion)
        
            return temp, temp_occlusion, tf.slice(tf.nn.softmax(temp),[0,0,0,1],[-1,-1,-1,-1]), tf.nn.softmax(temp_occlusion)
        
        else:
            temp = layers.conv2D_layer(bottom=temp,
                name='last_conv_temp',
                
                num_filters=num_classes,
                
                activation=linear_activation)
    
            return temp, temp_occlusion, tf.nn.sigmoid(temp), tf.nn.softmax(temp_occlusion)
    
    
def my_side_decoder(endpoints,y,reuse=False,num_classes=6, scope='side_decoder'):
    """Produces the mask from the encoder endpoints and the memory read output
    Args:
    endpoints: The output of the residual block in each stage of the encoder ()
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    
    """
    
    num_filters = 128
    
#    layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
    with tf.variable_scope(scope, reuse=reuse):
        y = layers.coord_conv2D_layer(bottom=y,
                name='init_conv',
                
                num_filters=num_filters,
                
                activation=linear_activation)
        with tf.variable_scope('init_res_block', reuse=reuse):
            y=deeplab_residual_block_RGMP(y, num_filters)
        
 
        for i in range(2):
            with tf.variable_scope('refine_module'+str(i), reuse=reuse):
                y = refinement_module(y, endpoints[i],num_filters)
            
        y_base = y
        
        with tf.variable_scope('occlusion_part', reuse=reuse):
            y=deeplab_residual_block_RGMP(y, num_filters)
        
   
        
        temp = layers.conv2D_layer(bottom=y,
                name='occlusion_last_conv1',
                
                num_filters=num_filters,
                
                activation=tf.nn.relu)
        temp_occlusion = layers.conv2D_layer(bottom=temp,
                name='occlusion_last_conv_temp',
                
                num_filters=2,
                
                activation=linear_activation)


        y=y_base
        
        with tf.variable_scope('mask_part', reuse=reuse):
            y=deeplab_residual_block_RGMP(y, num_filters)
        
        
   
        
        temp = layers.conv2D_layer(bottom=y,
                name='last_conv1',
                
                num_filters=num_filters,
                
                activation=tf.nn.relu)
        
        # if num_classes == 1:
        #     temp = layers.conv2D_layer(bottom=temp,
        #         name='last_conv_temp',
                
        #         num_filters=2,
                
        #         activation=linear_activation)
        #     return temp, temp_occlusion, tf.slice(tf.nn.softmax(temp),[0,0,0,1],[-1,-1,-1,-1]), tf.nn.softmax(temp_occlusion)
        # else:
        temp = layers.conv2D_layer(bottom=temp,
            name='last_conv_temp',
            
            num_filters=num_classes,
            
            activation=linear_activation)

        return temp, temp_occlusion, tf.nn.sigmoid(temp), tf.nn.softmax(temp_occlusion)


def my_image_decoder(endpoints, y,reuse=False,scope='my_image_decoder'):
    """Produces the mask from the encoder endpoints and the memory read output
    Args:
    endpoints: The output of the residual block in each stage of the encoder ()
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    
    """
    
    num_filters = 256
    
    with tf.variable_scope(scope, reuse=reuse):
        y = layers.conv2D_layer(bottom=y,
                name='init_conv',
                
                num_filters=num_filters,
                
                activation=linear_activation)
        with tf.variable_scope('init_res_block', reuse=reuse):
            y_base=deeplab_residual_block_RGMP(y, num_filters)
        
        y=y_base
        for i in range(2):
            with tf.variable_scope('refine_module'+str(i), reuse=reuse):
                y = refinement_module(y, endpoints[i],num_filters)
        
   
        
    return y

def resnet_feature_refiner(endpoints, y,reuse=False, scope='resnet_feature_refiner'):
    """Produces the mask from the encoder endpoints and the memory read output
    Args:
    endpoints: The output of the residual block in each stage of the encoder ()
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    
    """
    
    num_filters = 256
    
    with tf.variable_scope(scope, reuse=reuse):
        y = layers.conv2D_layer(bottom=y,
                name='init_conv',
                
                num_filters=num_filters,
                
                activation=linear_activation)
        with tf.variable_scope('init_res_block', reuse=reuse):
            y_base=residual_block_RGMP(y, num_filters)
        
        y=y_base
        for i in range(2):
            with tf.variable_scope('refine_module'+str(i), reuse=reuse):
                y = refinement_module(y, endpoints[i],num_filters)

    return y

def image_encoder(input_img,input_mask,my_model_options,downsample_stages=4,use_deeplab=True,is_training=True, reuse=False, scope='image_encoder'):
    with tf.variable_scope(scope, reuse=reuse):
        image_shape = input_img.get_shape().as_list()[1:3]
        logging.error('IMAGE SHAPE ' + str(image_shape))
   
        backbone_out, relative_end_points, endpoints = backbone_deeplab(input_img,input_mask,scope,my_model_options,is_training=is_training, reuse=reuse)
        backbone_shape = backbone_out.get_shape().as_list()
        logging.error('BACKBONE SHAPE ' + str(backbone_shape))
        
        
        if downsample_stages==2:
            
            backbone_out = xception_decoder(backbone_out,endpoints,image_shape,reuse=reuse)
        elif downsample_stages==3:
            return backbone_out, relative_end_points, endpoints
            
        else:
            backbone_out = tf.image.resize(
        backbone_out, [int(backbone_shape[1]/2),int(backbone_shape[2]/2)] ,method='bilinear',name='backbone_resize'  )
        
        logging.error('BACKBONE SHAPE ' + str(backbone_out))

    return backbone_out, relative_end_points, endpoints





def backbone_deeplab(input_img,input_mask,higher_scope,my_model_options, is_training=True, reuse=False,scope='mem_net_backbone'):
    """Defines the OSVOS network
    Args:
    inputs: Tensorflow placeholder that contains the input image
    scope: Scope name for the network
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """
    
    
#    image_size=(320,320)
#    relative_end_points = []


    with tf.variable_scope('mem_net_backbone', reuse=reuse):
        features, end_points = model.extract_features(
          input_img,
          my_model_options,
          is_training=False)
        # logging.error(str(end_points))
        # logging.error('FEATURES ' + str(features))
        
        modified_endpoints = dict()
        
        for entry in list(end_points.keys()):
            temp_entry = entry.replace(higher_scope + '/mem_net_backbone/', '')
            modified_endpoints[temp_entry] = end_points[entry]
            

        
    relative_endpoints=[]
    
    relative_endpoints.append(features)
    relative_endpoints.append(modified_endpoints['xception_65/entry_flow/block2/unit_1/xception_module/separable_conv2_pointwise'])
    

    return features, relative_endpoints,modified_endpoints

def xception_decoder(features,modified_endpoints,image_size,reuse=False):
    with tf.variable_scope('pretrained_decoder', reuse=reuse):
    
    
        upsampled_features = model.refine_by_decoder(features,
                  modified_endpoints,
                  crop_size=image_size,
                  decoder_output_stride=[4],
                  decoder_use_separable_conv=True,
                  decoder_use_sum_merge=False,
                  decoder_filters=256,
                  decoder_output_is_logits=False,
                  model_variant='xception_65',
                  weight_decay=0.0001,
                  reuse=None,
                  is_training=False,
                  fine_tune_batch_norm=False,
                  use_bounded_activation=False,
                  sync_batch_norm_method='None')
        
        return upsampled_features


def post_concat_block(features,exp_config,reuse=False,scope = 'post_concat_block'):
    num_filters = 256
    
    with tf.variable_scope(scope, reuse=reuse):
        y = layers.conv2D_layer(bottom=features,
                name='init_conv',
                
                num_filters=num_filters,
                
                activation=linear_activation)
        with tf.variable_scope('init_res_block', reuse=reuse):
            y=tf.nn.relu(deeplab_residual_block_RGMP(y, num_filters))
        


    return y
    


def deeplab_occlusion_finder(features,exp_config,reuse=False,scope = 'deeplab_occlusion_finder'):
    num_filters = 128
    
#    layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
    with tf.variable_scope(scope, reuse=reuse):
        y = layers.conv2D_layer(bottom=features,
                name='init_conv',
                
                num_filters=num_filters,
                
                activation=linear_activation)
        with tf.variable_scope('init_res_block', reuse=reuse):
            y=deeplab_residual_block_RGMP(y, num_filters)
        
#        with tf.variable_scope('second_res_block', reuse=reuse):
#            y=residual_block_RGMP(y, num_filters)
        
        temp = layers.coord_conv2D_layer(bottom=tf.nn.relu(y),
                name='last_conv1',
                
                num_filters=num_filters,
                
                activation=tf.nn.relu)


        temp = layers.conv2D_layer(bottom=temp,
                name='last_conv4',
                
                num_filters=2,
                
                activation=linear_activation)        

    return temp, tf.nn.softmax(temp)
    
def deeplab_refiner(features,exp_config,reuse=False,scope = 'deeplab_refiner'):
    num_filters = 128
    
#    layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']
    with tf.variable_scope(scope, reuse=reuse):
        y = layers.conv2D_layer(bottom=features,
                name='init_conv',
                
                num_filters=num_filters,
                
                activation=linear_activation)
        with tf.variable_scope('init_res_block', reuse=reuse):
            y=deeplab_residual_block_RGMP(y, num_filters)
        
#        with tf.variable_scope('second_res_block', reuse=reuse):
#            y=residual_block_RGMP(y, num_filters)
        
        temp = layers.coord_conv2D_layer(bottom=tf.nn.relu(y),
                name='last_conv1',
                
                num_filters=num_filters,
                
                activation=tf.nn.relu)


        temp = layers.conv2D_layer(bottom=temp,
                name='last_conv4',
                
                num_filters=exp_config.num_classes,
                
                activation=linear_activation)        

    return temp, tf.nn.sigmoid(temp)

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)



# Set deconvolutional layers to compute bilinear interpolation
def interp_surgery(variables):
    interp_tensors = []
    for v in variables:
        if 'upscale' in v.name:
            print('Variable in surgery : ' + str(v.name))
            if 'weights' in v.name:
                print('Variable in surgery : ' + str(v.name))
                h, w, k, m = v.get_shape()
                tmp = np.zeros((m, k, h, w))
                if m != k:
                    raise ValueError('input + output channels need to be the same')
                if h != w:
                    raise ValueError('filters need to be square')
                up_filter = upsample_filt(int(h))
                tmp[range(m), range(k), :, :] = up_filter
                interp_tensors.append(tf.assign(v, tmp.transpose((2, 3, 1, 0)), validate_shape=True, use_locking=True))
    return interp_tensors

def _mean_image_subtraction(image):
  """Subtracts the given means from each image channel.
  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)
  Note that the rank of `image` must be known.
  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
  Returns:
    the centered image.
  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  
  shapes = image.get_shape()
  means_tensor = tf.constant(
    np.array([123.68, 116.779, 103.939], dtype=np.single)
  )

  temp = tf.tile(means_tensor,[shapes[0],shapes[1],shapes[2],1])
  
  return image-temp

# TO DO: Move preprocessing into Tensorflow
def preprocess_img(image):
    """Preprocess the image to adapt it to network requirements
    Args:
    Image we want to input the network (W,H,3) numpy array
    Returns:
    Image ready to input the network (1,W,H,3)
    """
    if type(image) is not np.ndarray:
        image = np.array(Image.open(image), dtype=np.uint8)

    in_ = np.subtract(image, np.array((123.68, 116.779, 103.939), dtype=np.float32))
    # in_ = tf.subtract(tf.cast(in_, tf.float32), np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))

    return in_


# TO DO: Move preprocessing into Tensorflow
def preprocess_labels(label):
    """Preprocess the labels to adapt them to the loss computation requirements
    Args:
    Label corresponding to the input image (W,H) numpy array
    Returns:
    Label ready to compute the loss (1,W,H,1)
    """
    if type(label) is not np.ndarray:
        label = np.array(Image.open(label).split()[0], dtype=np.uint8)
    max_mask = np.max(label) * 0.5
    label = np.greater(label, max_mask)
    label = np.expand_dims(np.expand_dims(label, axis=0), axis=3)
    # label = tf.cast(np.array(label), tf.float32)
    # max_mask = tf.multiply(tf.reduce_max(label), 0.5)
    # label = tf.cast(tf.greater(label, max_mask), tf.float32)
    # label = tf.expand_dims(tf.expand_dims(label, 0), 3)
    return label


def load_vgg_imagenet(ckpt_path):
    """Initialize the network parameters from the VGG-16 pre-trained model provided by TF-SLIM
    Args:
    Path to the checkpoint
    Returns:
    Function that takes a session and initializes the network
    """
    reader = tf.train.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    vars_corresp = dict()
    for v in var_to_shape_map:
        if "conv" in v:
            vars_corresp[v] = slim.get_model_variables(v.replace("vgg_16", "osvos"))[0]
    init_fn = slim.assign_from_checkpoint_fn(
        ckpt_path,
        vars_corresp)
    return init_fn
