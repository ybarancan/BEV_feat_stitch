# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

#import nibabel as nib
import numpy as np
import os
import glob
import tensorflow as tf
import logging
import cv2
#import sys
#from skimage import measure, transform
import copy
import math
from PIL import Image
from pyquaternion import Quaternion


means_image = np.array([123.68, 116.779, 103.939], dtype=np.single)


def get_batch_size(inputs):
    return tf.cast(tf.shape(inputs)[0], tf.float32)
def flatten(tensor):
    '''
    Flatten the last N-1 dimensions of a tensor only keeping the first one, which is typically 
    equal to the number of batches. 
    Example: A tensor of shape [10, 200, 200, 32] becomes [10, 1280000] 
    '''
#    rhs_dim = get_rhs_dim(tensor)
    rhs_dim = tensor.get_shape().as_list()[-1]
    return tf.reshape(tensor, [-1, rhs_dim])

def get_rhs_dim(tensor):
    '''
    Get the multiplied dimensions of the last N-1 dimensions of a tensor. 
    I.e. an input tensor with shape [10, 200, 200, 32] leads to an output of 1280000 
    '''
    shape = tensor.get_shape().as_list()
    return np.prod(shape[1:])


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


def get_confusion(exp_config, annotation, segmentation,tot_mask,mask_iou=True):
    
    void_pixels = 1-tot_mask
    bg = (1 - np.clip(np.sum(np.float32(annotation[...,:exp_config.num_object_classes]),axis=-1),0,1)).astype(np.bool)
    
#    annotation = annotation.astype(np.bool)& void_pixels
#    segmentation = segmentation.astype(np.bool)& void_pixels
    annotation = annotation.astype(np.bool) 
    segmentation = segmentation.astype(np.bool)
    
    whole_confuse = []
    for k in range(exp_config.num_object_classes+1):
        temp_confuse = []
        cur_est = segmentation[...,k]
        for m in range(exp_config.num_object_classes):
            temp_confuse.append(np.float32(np.sum((cur_est & annotation[...,m])& void_pixels)))
            
        temp_confuse.append(np.float32(np.sum((cur_est & bg)& void_pixels)))
            
        whole_confuse.append(np.squeeze(np.array(temp_confuse)))
    
    return np.array(whole_confuse)


        
def argoverse_project_to_image(exp_config, bev_image, calib_ref):
    
    sample_point1 = np.array([int(3*bev_image.shape[1]/8),int(2*bev_image.shape[0]/8),0],np.float32)
    sample_point2 = np.array([int(5*bev_image.shape[1]/8),int(2*bev_image.shape[0]/8),0],np.float32)
    sample_point3 = np.array([int(3*bev_image.shape[1]/8),int(6*bev_image.shape[0]/8),0],np.float32)
    sample_point4 = np.array([int(5*bev_image.shape[1]/8),int(6*bev_image.shape[0]/8),0],np.float32)
    
    sample_points = np.stack([sample_point1,sample_point2,sample_point3,sample_point4],axis=-1)
    
    source_points = np.float32(np.copy(sample_points[0:2,:]).T)
    
    sample_points[1,:] = bev_image.shape[0] - sample_points[1,:]
    
    sample_points[0,:] = -(sample_points[0,:] - bev_image.shape[1]/2)*exp_config.resolution
    sample_points[1,:] = sample_points[1,:]*exp_config.resolution
    
    # sample_points[2,:]
    sample_points=np.stack([sample_points[1,:],sample_points[0,:],sample_points[2,:]])

    cor_points = calib_ref.project_ego_to_image(sample_points.T)


    # logging.error('COR POINTS ' + str(cor_points))

    
    estimated_transform = cv2.getPerspectiveTransform(source_points,np.float32(cor_points[:,:2]))
    # modified_transform=estimated_transform
    # total_image_size = (1920,1200)
    # warped_label = cv2.warpPerspective(bev_image,modified_transform,total_image_size,flags=cv2.INTER_LINEAR)
    

    
    return estimated_transform     


def argoverse_project_to_ground(exp_config, image1,label,calib_ref,pose_ref,calib_cur,pose_cur,cam_intrinsic,reference_frame=False):
    # extents = exp_config.extents
    # resolution = exp_config.resolution
    # image1 = image1*vis_mask
#    image2 = image2*vis_mask

    # covered_region = np.ones_like(image1) * vis_mask
    
    extents = exp_config.extents
    resolution = exp_config.resolution

  

    total_image_size = exp_config.total_image_size

    # Get rasterised map
    
    sample_point1 = np.array([int(3*image1.shape[1]/8),int(7*image1.shape[0]/8),1],np.float32)
    sample_point2 = np.array([int(5*image1.shape[1]/8),int(7*image1.shape[0]/8),1],np.float32)
    sample_point3 = np.array([int(3*image1.shape[1]/8),int(6*image1.shape[0]/8),1],np.float32)
    sample_point4 = np.array([int(5*image1.shape[1]/8),int(6*image1.shape[0]/8),1],np.float32)
    
    base_sample_points = np.stack([sample_point1,sample_point2,sample_point3,sample_point4],axis=-1)
    
    source_points = np.float32(base_sample_points[0:2,:].T)
    
    
    
    if reference_frame:
        camera_height = calib_ref.T[1]
        sample_points = np.copy(base_sample_points)
        sample_points[-1,:] = cam_intrinsic[0,0]*camera_height/(np.abs(base_sample_points[1,:]-cam_intrinsic[1,-1]))
        sample_points = sample_points.T
    #First project image to ego    
        cor_cam_points = np.float32(calib_ref.project_image_to_cam(sample_points))
        
#        to_estimate_cor_cam_points = cor_cam_points[:,:2]
        
        to_estimate_cor_cam_points = np.zeros((4,2),np.float32)
        to_estimate_cor_cam_points[:,0] = np.float32(cor_cam_points[:,0])
        to_estimate_cor_cam_points[:,1] = np.float32(cor_cam_points[:,2])
        
        corresponding_points = np.copy(to_estimate_cor_cam_points)
        corresponding_points[:,0] = (corresponding_points[:,0] + extents[1])/resolution
        my_ys = corresponding_points[:,1]
        my_ys = my_ys - extents[2]
        my_ys = (extents[3] - extents[2])/resolution - my_ys/resolution
        corresponding_points[:,1] = my_ys
        
        
        estimated_transform = cv2.getPerspectiveTransform(source_points,corresponding_points)
        
        modified_transform = estimated_transform
        translation_matrix = np.eye(3)
        
        extra_space = exp_config.extra_space
        
        translation_matrix[0,-1] = extra_space[0]/2
        translation_matrix[1,-1] = extra_space[1]/2
        
        modified_transform = np.dot(translation_matrix, modified_transform)
        warped_image_ref= cv2.warpPerspective(image1,modified_transform,total_image_size,flags=cv2.INTER_LINEAR)
        # warped_covered = cv2.warpPerspective(covered_region,modified_transform,total_image_size,flags=cv2.INTER_LINEAR)
        warped_label = cv2.warpPerspective(label,modified_transform,total_image_size,flags=cv2.INTER_NEAREST)
        
        return warped_image_ref, warped_label, modified_transform
        
        
    else:
        
        camera_height = calib_ref.T[1]
        sample_points = np.copy(base_sample_points)
        sample_points[-1,:] = cam_intrinsic[0,0]*camera_height/(np.abs(base_sample_points[1,:]-cam_intrinsic[1,-1]))
        sample_points = sample_points.T
    #First project image to ego    
        cor_ego_points = np.float32(calib_cur.project_image_to_ego(sample_points))
        temp_points = np.ones((4,4),np.float32)
        temp_points[:,:3] = cor_ego_points
        cor_map_points = np.matmul(pose_cur,temp_points.T)
        
        cor_new_ego_points = np.matmul(np.linalg.inv(pose_ref),cor_map_points)
        
        cor_cam_points = calib_ref.project_ego_to_cam(cor_new_ego_points.T[:,:3])
        
        to_estimate_cor_cam_points = np.zeros((4,2),np.float32)
        to_estimate_cor_cam_points[:,0] = np.float32(cor_cam_points[:,0])
        to_estimate_cor_cam_points[:,1] = np.float32(cor_cam_points[:,2])
        
        corresponding_points = np.copy(to_estimate_cor_cam_points)
        corresponding_points[:,0] = (corresponding_points[:,0] + extents[1])/resolution
        my_ys = corresponding_points[:,1]
        my_ys = my_ys - extents[2]
        my_ys = (extents[3] - extents[2])/resolution - my_ys/resolution
        corresponding_points[:,1] = my_ys
        
        
        estimated_transform = cv2.getPerspectiveTransform(source_points,corresponding_points)
        
        modified_transform = estimated_transform
        translation_matrix = np.eye(3)
        
        extra_space = exp_config.extra_space
        
        translation_matrix[0,-1] = extra_space[0]/2
        translation_matrix[1,-1] = extra_space[1]/2
        
        modified_transform = np.dot(translation_matrix, modified_transform)
        warped_image_dest = cv2.warpPerspective(image1,modified_transform,total_image_size,flags=cv2.INTER_LINEAR)
        # warped_covered = cv2.warpPerspective(covered_region,modified_transform,total_image_size,flags=cv2.INTER_LINEAR)
        warped_label = cv2.warpPerspective(label,modified_transform,total_image_size,flags=cv2.INTER_NEAREST)
        
        return warped_image_dest,  warped_label, modified_transform 
    


def argoverse_tensorflow_project_to_ground(exp_config, image1,source_image1,pose_ref, calib_ref,pose_cur,calib_cur, cam_intrinsic,reference_frame=False):
    
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
    

    if reference_frame:
        camera_height = calib_ref.T[1]
        sample_points = np.copy(base_sample_points)
        sample_points[-1,:] = cam_intrinsic[0,0]*camera_height/(np.abs(base_sample_points[1,:]-cam_intrinsic[1,-1]))
        sample_points = sample_points.T
    #First project image to ego    
        cor_cam_points = np.float32(calib_ref.project_image_to_cam(sample_points))
        
#        to_estimate_cor_cam_points = cor_cam_points[:,:2]
        
        to_estimate_cor_cam_points = np.zeros((4,2),np.float32)
        to_estimate_cor_cam_points[:,0] = np.float32(cor_cam_points[:,0])
        to_estimate_cor_cam_points[:,1] = np.float32(cor_cam_points[:,2])
        
        corresponding_points = np.copy(to_estimate_cor_cam_points)
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
        camera_height = calib_ref.T[1]
        sample_points = np.copy(base_sample_points)
        sample_points[-1,:] = cam_intrinsic[0,0]*camera_height/(np.abs(base_sample_points[1,:]-cam_intrinsic[1,-1]))
        sample_points = sample_points.T
    #First project image to ego    
        cor_ego_points = np.float32(calib_cur.project_image_to_ego(sample_points))
        temp_points = np.ones((4,4),np.float32)
        temp_points[:,:3] = cor_ego_points
        cor_map_points = np.matmul(pose_cur,temp_points.T)
        
        cor_new_ego_points = np.matmul(np.linalg.inv(pose_ref),cor_map_points)
        
        cor_cam_points = calib_ref.project_ego_to_cam(cor_new_ego_points.T[:,:3])
        
        to_estimate_cor_cam_points = np.zeros((4,2),np.float32)
        to_estimate_cor_cam_points[:,0] = np.float32(cor_cam_points[:,0])
        to_estimate_cor_cam_points[:,1] = np.float32(cor_cam_points[:,2])
        
        corresponding_points = np.copy(to_estimate_cor_cam_points)
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
    
    

def tensorflow_project_to_ground(exp_config, image1,source_image1,pose1, cs1,pose2,cs2, cam_intrinsic,reference_frame=False,grid=False):
    
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
    
    inverse_pr2 = np.linalg.inv(pr2)
    pt = np.array(pose1['translation']).reshape((-1, 1))
    pt2 = np.array(pose2['translation']).reshape((-1, 1))
    
    cr = Quaternion(cs1['rotation']).rotation_matrix.T
    
    inverse_cr = np.linalg.inv(cr)

    
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
    
def project_to_ground(exp_config, image1,label1,pose1, cs1,pose2,cs2, cam_intrinsic,vis_mask,reference_frame=False,grid=False):
    
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
    
    inverse_pr2 = np.linalg.inv(pr2)
    pt = np.array(pose1['translation']).reshape((-1, 1))
    pt2 = np.array(pose2['translation']).reshape((-1, 1))
    
    cr = Quaternion(cs1['rotation']).rotation_matrix.T
    
    inverse_cr = np.linalg.inv(cr)

    
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

def tensorflow_project_bev_to_bev(nusc, exp_config, cur_sample, next_sample):
    
    camera_channel='CAM_FRONT'
    cam_token_ref = next_sample['data'][camera_channel]
    cam_record_ref = nusc.get('sample_data', cam_token_ref)
    
    
    pose1 = nusc.get('ego_pose', cam_record_ref['ego_pose_token'])
    
    '''
    '''
    cam_token_cur = cur_sample['data'][camera_channel]
    cam_record_cur = nusc.get('sample_data', cam_token_cur)
    
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
    
    pr = Quaternion(pose1['rotation']).rotation_matrix.T
    pr2 = Quaternion(pose2['rotation']).rotation_matrix.T
    
    inverse_pr2 = np.linalg.inv(pr2)
    pt = np.array(pose1['translation']).reshape((-1, 1))
    pt2 = np.array(pose2['translation']).reshape((-1, 1))
    
    
    
    coef_matrix = inverse_pr2
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
    
def project_bev_to_bev(nusc, exp_config, cur_sample, next_sample, cur_label, next_label):
    
    camera_channel='CAM_FRONT'
    cam_token_ref = next_sample['data'][camera_channel]
    cam_record_ref = nusc.get('sample_data', cam_token_ref)
   
    pose1 = nusc.get('ego_pose', cam_record_ref['ego_pose_token'])
    
    '''
    '''
    cam_token_cur = cur_sample['data'][camera_channel]
    cam_record_cur = nusc.get('sample_data', cam_token_cur)
   
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
    
    pr = Quaternion(pose1['rotation']).rotation_matrix.T
    pr2 = Quaternion(pose2['rotation']).rotation_matrix.T
    
    inverse_pr2 = np.linalg.inv(pr2)
    pt = np.array(pose1['translation']).reshape((-1, 1))
    pt2 = np.array(pose2['translation']).reshape((-1, 1))
    
    
    
    coef_matrix = inverse_pr2
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

def iou_calculator(annotation, segmentation,vis_mask,occ_mask,mask_iou=False, void_pixels=None):
    """
    annotation : gt mask
    segmentation : method estimate
    """
#   
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



def train_saver(exp_config, log_dir, batch_image, batch_label, batch_dilated_label):
    
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

def project_to_image(exp_config, image1, cs1,cam_intrinsic):
   
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




def binary_to_png_convert(exp_config, label):
    
    label = np.squeeze(label)
    label_creator_array = np.ones((label.shape[0],label.shape[1],int(exp_config.num_classes)),np.float32)
    
    for k in range(int(exp_config.num_classes)):
        label_creator_array[...,k] = 2**(k+1)
        
    png_label = np.uint8(np.squeeze(np.sum(label*label_creator_array,axis=-1)))
    return png_label


def png_to_binary(cropped_label, total_label_slices):
    temp_label = np.ones((cropped_label.shape[0],cropped_label.shape[1],int(total_label_slices )))
    
    rem = np.copy(cropped_label)
    for k in range(total_label_slices ):
        temp_rem = rem//(2**int(total_label_slices -k-1))
#        logging.error('TEMP REM SHAPE : ' + str(temp_rem.shape))
        
        temp_label[:,:,int(total_label_slices -k-1)] = np.copy(temp_rem)
        
        rem = rem%(2**int(total_label_slices -k-1))
    return temp_label


def png_to_binary_with_ones(exp_config, cropped_label):
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
def get_image_and_label(target_dir, my_scene, ind):
    
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
    

def get_visible_mask(instrinsics, image_width, extents, resolution):

    # Get calibration parameters
    fu, cu = instrinsics[0, 0], instrinsics[0, 2]

    # Construct a grid of image coordinates
    x1, z1, x2, z2 = extents
    x, z = np.arange(x1, x2, resolution), np.arange(z1, z2, resolution)
    ucoords = x / z[:, None] * fu + cu

    # Return all points which lie within the camera bounds
    return (ucoords >= 0) & (ucoords < image_width)

#
#
#
#def save_array(array,name,slice_last_dim=True,is_rgb=False,to_size=None,correct=True,val=False):
#     
#    
#        
#     if is_rgb:
#         if correct:
#             if not use_deeplab:
#                 array = array + means_image
#     for k in range(array.shape[0]):
#        
#         cur_slice = np.squeeze(array[k,...])
#         if to_size is not None:
#             if is_rgb:
#                 cur_slice = cv2.resize(cur_slice,to_size, interpolation = cv2.INTER_LINEAR)
#             else:
#                 cur_slice = cv2.resize(cur_slice,to_size, interpolation = cv2.INTER_NEAREST)
#         
#         if is_rgb:
#             img_png=Image.fromarray(np.uint8(cur_slice))
#             
#             if val:
#                 img_png.save(os.path.join(validation_res_path,name+'_'+str(k)+'.jpg'))
#             else:
#                img_png.save(os.path.join(train_results_path,name+'_'+str(k)+'.jpg'))
#         else:
#             
#             if slice_last_dim:
#                 for m in range(cur_slice.shape[-1]):
#                     
#                     img_png=Image.fromarray(np.uint8(255*cur_slice[...,m]))
#                     if val:
#                        img_png.save(os.path.join(validation_res_path,name+'_batch_'+str(k)+'_class_'+str(m)+'.jpg'))
#                     else:
#                        img_png.save(os.path.join(train_results_path,name+'_batch_'+str(k)+'_class_'+str(m)+'.jpg'))
#                     
#             else:
#                 
#                 
#                img_png=Image.fromarray(np.uint8(cur_slice*255))
#                if val:
#                    img_png.save(os.path.join(validation_res_path,name+'_'+str(k)+'.jpg'))
#                else:
#                    img_png.save(os.path.join(train_results_path,name+'_'+str(k)+'.jpg'))
         
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

def write_results_to_folder(validation_res_path, name_of_seq,frame_number,to_eval_estimates):
    
    root_folder = os.path.join(validation_res_path,name_of_seq)
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    len_of_number = len(str(frame_number))
    init_str = str(frame_number)
    for k in range(5-len_of_number):
        init_str = '0'+init_str
    img_png=Image.fromarray(to_eval_estimates.astype(np.uint8))
    img_png.save(os.path.join(root_folder,init_str+'.png'))    
    
def  single_image_inception_preprocess(exp_config, total_label_slices, use_deeplab, orig_img, orig_label):
    
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


def get_label_mask(exp_config, ar):
    
    vis_mask = ar[...,exp_config.num_classes]
    occ_mask = ar[...,exp_config.num_classes+1]
    
    tot_mask = occ_mask*vis_mask
    
    return tot_mask


    
def expand_image(img,new_sizes,left_up):
    
    if len(img.shape) ==3:
    
        new_img = np.zeros((new_sizes[0],new_sizes[1],img.shape[2]),img.dtype)
        new_img[int(left_up[0]):int(left_up[0]+img.shape[0]),int(left_up[1]):int(left_up[1]+img.shape[1]),:] = img
    else:
        new_img = np.zeros((new_sizes[0],new_sizes[1]),img.dtype)
        new_img[int(left_up[0]):int(left_up[0]+img.shape[0]),int(left_up[1]):int(left_up[1]+img.shape[1])] = img
    
    
    return new_img
