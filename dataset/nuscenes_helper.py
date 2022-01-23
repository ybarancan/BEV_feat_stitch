#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:32:43 2019

@author: cany
"""

from PIL import Image
import numpy as np
import logging
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box
from pyquaternion import Quaternion
import cv2

#from geomath.hulls import ConcaveHull
import os

from skimage.draw import polygon as skimage_polygon


global_const = 3.99303084


def get_whole_mask(cur_list, img_size):
    
    mask = np.ones(img_size,np.uint8)
    
    for k in range(len(cur_list)):
        temp_mask = 1-get_mask(cur_list[k],img_size)
        mask = mask*temp_mask
        
    return 1-mask

def get_mask(corners,img_size):
    
    
    corners_ar = np.array(corners)
    
    
    
    corners_ar = corners_ar[:,::-1]
     
    mask = np.zeros(img_size,np.uint8)
    
#    logging.error('About to get the polygon')
    
    rr, cc = skimage_polygon(corners_ar[:,0], corners_ar[:,1],shape=tuple(img_size))
    
    
#    logging.error('Polygon obtained')
    mask[rr, cc] = 1


    return mask



def get_image_and_mask(nusc,sample_token, map_api,layer_names = ['drivable_area','ped_crossing', 'walkway', 'carpark_area','road_segment','lane'],camera_channel = 'CAM_FRONT' ):
    
    np_img, vertices_per_layer, my_lidar_mask = render_map_in_image(nusc, map_api, sample_token, layer_names=layer_names, camera_channel=camera_channel)

#    logging.error('Image and map are rendered')

    total_mask = np.zeros((np_img.shape[0],np_img.shape[1],len(layer_names)+2),np.uint8)
    
    for k in range(len(layer_names)):
        layer_name = layer_names[k]
        cur_list = vertices_per_layer[layer_name]
        if len(cur_list) < 1:
            layer_mask = np.zeros((np_img.shape[0],np_img.shape[1]),np.uint8)
        else:
            layer_mask = get_whole_mask(cur_list,np_img.shape[0:2])
        total_mask[:,:,k] = layer_mask
        
    layer_name = 'visible'
    cur_list = vertices_per_layer[layer_name]
    if len(cur_list) < 1:
        layer_mask = np.zeros((np_img.shape[0],np_img.shape[1]),np.uint8)
    else:
        layer_mask = get_whole_mask(cur_list,np_img.shape[0:2])
    total_mask[:,:,len(layer_names)] = layer_mask
        
    total_mask[:,:,-1] = my_lidar_mask
    return np.float32(np_img), total_mask    

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


color_map = dict(drivable_area='#a6cee3',
                             road_segment='#1f78b4',
                             road_block='#b2df8a',
                             lane='#33a02c',
                             ped_crossing='#fb9a99',
                             walkway='#e31a1c',
                             stop_line='#fdbf6f',
                             carpark_area='#ff7f00',
                             road_divider='#cab2d6',
                             lane_divider='#6a3d9a',
                             traffic_light='#7e772e',
                             occlusion='#7e772e')


def render_polygon(mask, polygon, extents, resolution, value=1):
    polygon = (polygon - np.array(extents[:2])) / resolution
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, value)


def render_shapely_polygon(mask, polygon, extents, resolution):

    if polygon.geom_type == 'Polygon':

        # Render exteriors
        render_polygon(mask, polygon.exterior.coords, extents, resolution, 1)

        # Render interiors
        for hole in polygon.interiors:
            render_polygon(mask, hole.coords, extents, resolution, 0)
    
    # Handle the case of compound shapes
    else:
        for poly in polygon:
            render_shapely_polygon(mask, poly, extents, resolution)


    
def expand_image(img,new_sizes,left_up):
    
    new_img = np.zeros((new_sizes[0],new_sizes[1],3),np.float32)
    new_img[int(left_up[0]):int(left_up[0]+img.shape[0]),int(left_up[1]):int(left_up[1]+img.shape[1]),:] = img
    return new_img
    

    
def render_map_in_image(    nusc, map_api, sample_token,layer_names,
                            camera_channel: str = 'CAM_FRONT',
                            alpha: float = 0.3,
                            patch_radius: float = 1000,
                            min_polygon_area: float = 1000,
                            render_behind_cam: bool = True,
                            render_outside_im: bool = True,
                          
                            verbose: bool = True,
                            out_path: str = None) -> None:
    
        
        near_plane = 1
        min_dist= 1.0
        lidar_dist_thresh=1
        lidar_height_thresh = 1
        apply_lidar_aligment=True
        vertices_per_layer = dict()
        if verbose:
            print('Warning: Note that the projections are not always accurate as the localization is in 2d.')

        # Default layers.
      
#        layer_names = ['drivable_area']
#        layer_names = ['walkway','drivable_area']
        
        

        # Check layers whether we can render them.
        for layer_name in layer_names:
            assert layer_name in map_api.non_geometric_polygon_layers, \
                'Error: Can only render non-geometry polygons: %s' % layer_names

        # Check that NuScenesMap was loaded for the correct location.
        sample_record = nusc.get('sample', sample_token)
        scene_record = nusc.get('scene', sample_record['scene_token'])
        log_record = nusc.get('log', scene_record['log_token'])
        log_location = log_record['location']
        assert map_api.map_name == log_location, \
            'Error: NuScenesMap loaded for location %s, should be %s!' % (map_api.map_name, log_location)

        # Grab the front camera image and intrinsics.
        cam_token = sample_record['data'][camera_channel]
        cam_record = nusc.get('sample_data', cam_token)
        cam_path = nusc.get_sample_data_path(cam_token)
        im = Image.open(cam_path)
#        im = Image.open(cam_path)
        np_img = np.array(im,np.uint8)
        
        im_size = im.size
        cs_record = nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])

        # Retrieve the current map.
        poserecord = nusc.get('ego_pose', cam_record['ego_pose_token'])
        ego_pose = poserecord['translation']
        box_coords = (
            ego_pose[0] - patch_radius,
            ego_pose[1] - patch_radius,
            ego_pose[0] + patch_radius,
            ego_pose[1] + patch_radius,
        )
        records_in_patch = map_api.get_records_in_patch(box_coords, layer_names, 'intersect')

        # Init axes.
#        fig = plt.figure(figsize=(9, 16))
#        ax = fig.add_axes([0, 0, 1, 1])
#        ax.set_xlim(0, im_size[0])
#        ax.set_ylim(0, im_size[1])
#        ax.imshow(im)
#        ax.invert_yaxis()
        extents = [-25., 1., 25., 50.]
        resolution=0.25
        
        map_patch = box(*extents)
    
        my_cam_trans = np.array(cs_record['translation']).reshape((-1, 1))
        
        patch_points = np.array(map_patch.exterior.xy)
        patch_points = np.vstack((patch_points[0,...],np.ones((1, patch_points.shape[1]))*my_cam_trans[2], patch_points[1,...],))
        
      
        points, norm_const = view_points(patch_points, cam_intrinsic, normalize=True)
    
        # Skip polygons where all points are outside the image.
        # Leave a margin of 1 pixel for aesthetic reasons.
        inside = np.ones(points.shape[1], dtype=bool)
        inside = np.logical_and(inside, points[0, :] > 1)
        inside = np.logical_and(inside, points[0, :] < im_size[0] - 1)
        inside = np.logical_and(inside, points[1, :] > 1)
        inside = np.logical_and(inside, points[1, :] < im_size[1] - 1)
      
    
        points = points[:2, :]
        points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
#        label = 'visible'
#        polygon_proj = Polygon(points)
#        ax.add_patch(descartes.PolygonPatch(polygon_proj, fc=color_map[layer_name], alpha=alpha,
#                                                                    label=label))
        
        layer_list=[]
        layer_list.append(np.array(points))
        vertices_per_layer['visible'] = layer_list
        
        patch_points = np.array(map_patch.exterior.xy)
        patch_points = np.vstack((patch_points[0,...], np.zeros((1, patch_points.shape[1])),patch_points[1,...]))
        
        warped_points = np.dot(np.linalg.inv(Quaternion(cs_record['rotation']).rotation_matrix.T), patch_points)
        
        my_cam_trans = np.array(cs_record['translation']).reshape((-1, 1))
#                    my_cam_trans[-1] = 0
        warped_points = warped_points + my_cam_trans
        warped_points[2,...] = 0
        
        warped_points = np.dot(np.linalg.inv(Quaternion(poserecord['rotation']).rotation_matrix.T), warped_points)
        warped_points = warped_points + np.array(poserecord['translation']).reshape((-1, 1))
        
        warped_poly_points = [(p0, p1) for (p0, p1) in zip(warped_points[0], warped_points[1])]
        warped_polygon_proj = Polygon(warped_poly_points)
        

        
        my_pc_points, bev_lidar_points = lidar_func(sample_record,nusc)
        bev_lidar_points = bev_lidar_points[:,bev_lidar_points[2,:] < lidar_height_thresh]
        cropped_lidar_points = my_pc_points[:,my_pc_points[2,:] > 0]
#        occluded = get_occlusion_mask(np.transpose(my_pc_points,[1,0]), extents, resolution)
        selected_lidar = get_occlusion_mask(cropped_lidar_points, extents, resolution,np.squeeze(np.array(cs_record['translation']).reshape((-1, 1))[-1]))
        points,_ = view_points(selected_lidar[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
        
        
        
        
        depths=selected_lidar[2,:]
#        min_dist
        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        
#        intensities = selected_lidar[3, :]
#        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
#        intensities = intensities ** 0.1
#        intensities = np.maximum(0, intensities - 0.5)
#        coloring = intensities
#        dot_size = 5
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        
        my_lidar_mask = np.zeros((im_size[1],im_size[0]))
        lidar_mask_res = 50
        for k in range(points.shape[1]):
            row_start = int(points[1,k]//lidar_mask_res)*lidar_mask_res
            row_end = np.clip(int((points[1,k]//lidar_mask_res) + 1)*lidar_mask_res,0,im_size[1])
            
            col_start = int(points[0,k]//lidar_mask_res)*lidar_mask_res
            col_end = np.clip(int((points[0,k]//lidar_mask_res) + 1)*lidar_mask_res,0,im_size[0])
            
            my_lidar_mask[row_start:row_end,col_start:col_end] = 1
            
        
        my_lidar_mask[-200:,int(im_size[0]/2 - 100):int(im_size[0]/2 + 100)] = 1
        
#        coloring = coloring[mask]
        
        
#        plt.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
        

        # Retrieve and render each record.
        
        
        lidar_poly_points=[]
        for k in range(bev_lidar_points.shape[1]):
            lid_po = Point(bev_lidar_points[0,k], bev_lidar_points[1,k])
            lidar_poly_points.append(lid_po)
            
        for layer_name in layer_names:
         
            layer_list=[]
            for token in records_in_patch[layer_name]:
                
                
                
                    record = map_api.get(layer_name, token)
                    if layer_name == 'drivable_area':
                        polygon_tokens = record['polygon_tokens']
                    else:
                        polygon_tokens = [record['polygon_token']]
    
                    for polygon_token in polygon_tokens:
                        
                        try:
                            polygon = map_api.extract_polygon(polygon_token)
                            
#                            map_patch = box(*extents)
#                            patch_points = np.array(map_patch.exterior.xy)
#                            patch_points = np.vstack((patch_points[0,...], np.zeros((1, patch_points.shape[1])),patch_points[1,...]))
#                            
#                            warped_points = np.dot(np.linalg.inv(Quaternion(cs_record['rotation']).rotation_matrix.T), patch_points)
#                            
#                            my_cam_trans = np.array(cs_record['translation']).reshape((-1, 1))
#        #                    my_cam_trans[-1] = 0
#                            warped_points = warped_points + my_cam_trans
#                            warped_points[2,...] = 0
#                            
#                            warped_points = np.dot(np.linalg.inv(Quaternion(poserecord['rotation']).rotation_matrix.T), warped_points)
#                            warped_points = warped_points + np.array(poserecord['translation']).reshape((-1, 1))
#                            
#                            warped_poly_points = [(p0, p1) for (p0, p1) in zip(warped_points[0], warped_points[1])]
#                            warped_polygon_proj = Polygon(warped_poly_points)
                            
                            if hasattr(polygon, 'geoms'):
                                logging.error("POLYGON HAS GEOMS " + str(hasattr(polygon, 'geoms')))
                            if hasattr(warped_polygon_proj, 'geoms'):
                                logging.error("WARPED POLYGON HAS GEOMS " + str(hasattr(warped_polygon_proj, 'geoms')))
                            
                            
                            if warped_polygon_proj.intersects(polygon):
        #                        break
                                inter = warped_polygon_proj.intersection(polygon)
#                                inter=polygon
                                
                                if hasattr(inter, 'geoms'):
                                    for temp_poly in inter.geoms:
#                                        
#                                        
#                                        
                                        
                                        points = np.array(temp_poly.exterior.xy)
                                        
                                        if apply_lidar_aligment:
                                            heights_list = []
                                            selected_lidars_list=[]
                                            for k in range(len(lidar_poly_points)):
                                                lid_po = lidar_poly_points[k]
                                                if temp_poly.contains(lid_po):
                                                    selected_height = bev_lidar_points[2,k]
                                                   
                                                    heights_list.append(selected_height)
                                                    selected_lidars_list.append(np.squeeze(bev_lidar_points[:,k]))
                #                                
                #                                lidar_dist = np.square(points[0,k]-bev_lidar_points[0,:])+np.square(points[1,k]-bev_lidar_points[1,:])
                #                                
                #                                nearest_lidar = np.argmin(lidar_dist)
                                                
                #                                nearest_dist = lidar_dist[nearest_lidar]
                #                                if nearest_dist < lidar_dist_thresh:
                #                                    selected_height = bev_lidar_points[2,nearest_lidar]
                #                                    heights_list.append(selected_height)
                                                
                                                
                #                                heights[:,k]=selected_height
                                            
                                            if len(heights_list) > 0:
                                                heights = np.ones((1,points.shape[1]))
                                                selected_lidars = np.array(selected_lidars_list)
                                                for k in range(points.shape[1]):
                                                    lidar_dist = np.square(points[0,k]-selected_lidars[:,0])+np.square(points[1,k]-selected_lidars[:,1])
                #                                
                                                    nearest_lidar = np.argmin(lidar_dist)
                                                
                                                 
                                                    selected_height = heights_list[nearest_lidar]
                                                 
                                                
                                                    heights[:,k]=selected_height
                                            
                                                
                                                
        #                                        heights = np.ones((1,points.shape[1]))*np.mean(heights_list)
                                            else:
                                                continue
                                        else:
                                            heights = np.zeros((1, points.shape[1]))
        #                                points = np.vstack((points, np.zeros((1, points.shape[1]))))
                                        points = np.vstack((points, heights))
                    
                                        # Transform into the ego vehicle frame for the timestamp of the image.
                                        points = points - np.array(poserecord['translation']).reshape((-1, 1))
                                        points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)
                    
                                        # Transform into the camera.
                                        points = points - np.array(cs_record['translation']).reshape((-1, 1))
                                        points = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points)
                                        
                                        
                                        
                                        camera_points = np.copy(points)
                                        
                                        # Remove points that are partially behind the camera.
                                        depths = points[2, :]
                    #                    behind = ((depths < near_plane) | (depths > 25*np.sqrt(2)))
                                        behind = depths < near_plane
                                        if np.all(behind):
                                            continue
                    
                                        if render_behind_cam:
                                            # Perform clipping on polygons that are partially behind the camera.
                                            points,_ = _clip_points_behind_camera(points, near_plane)
                                        elif np.any(behind):
                                            # Otherwise ignore any polygon that is partially behind the camera.
                                            continue
                    
                                        # Ignore polygons with less than 3 points after clipping.
                                        if len(points) == 0 or points.shape[1] < 3:
                                            continue
                    
                                        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                                        points, norm_const = view_points(points, cam_intrinsic, normalize=True)
                    
                                        # Skip polygons where all points are outside the image.
                                        # Leave a margin of 1 pixel for aesthetic reasons.
                                        inside = np.ones(points.shape[1], dtype=bool)
                                        inside = np.logical_and(inside, points[0, :] > 1)
                                        inside = np.logical_and(inside, points[0, :] < im.size[0] - 1)
                                        inside = np.logical_and(inside, points[1, :] > 1)
                                        inside = np.logical_and(inside, points[1, :] < im.size[1] - 1)
                                        if render_outside_im:
                                            if np.all(np.logical_not(inside)):
                                                continue
                                        else:
                                            if np.any(np.logical_not(inside)):
                                                continue
                    
                                        points = points[:2, :]
                                        points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
                                        polygon_proj = Polygon(points)
        #                                polygon_proj = polygon_proj.intersection(non_occ_polygon_proj)
                                        # Filter small polygons
                                        if polygon_proj.area < min_polygon_area:
                                            continue
                                        layer_list.append(np.array(points))
        #                                label = layer_name
        #                                ax.add_patch(descartes.PolygonPatch(polygon_proj, fc=color_map[layer_name], alpha=alpha,
        #                                                                    label=label))
                                        
                                else:
                                    points = np.array(inter.exterior.xy)
                                    
        #                            heights = np.zeros((1,points.shape[1]),np.float32)
                                    
                                    if apply_lidar_aligment:
                                        heights_list = []
                                        selected_lidars_list=[]
                                        for k in range(len(lidar_poly_points)):
                                            lid_po = lidar_poly_points[k]
                                            if inter.contains(lid_po):
                                                selected_height = bev_lidar_points[2,k]
                                               
                                                heights_list.append(selected_height)
                                                selected_lidars_list.append(np.squeeze(bev_lidar_points[:,k]))
            #                                
            #                                lidar_dist = np.square(points[0,k]-bev_lidar_points[0,:])+np.square(points[1,k]-bev_lidar_points[1,:])
            #                                
            #                                nearest_lidar = np.argmin(lidar_dist)
                                            
            #                                nearest_dist = lidar_dist[nearest_lidar]
            #                                if nearest_dist < lidar_dist_thresh:
            #                                    selected_height = bev_lidar_points[2,nearest_lidar]
            #                                    heights_list.append(selected_height)
                                            
                                            
            #                                heights[:,k]=selected_height
                                        
                                        if len(heights_list) > 0:
                                            heights = np.ones((1,points.shape[1]))
                                            selected_lidars = np.array(selected_lidars_list)
                                            for k in range(points.shape[1]):
                                                lidar_dist = np.square(points[0,k]-selected_lidars[:,0])+np.square(points[1,k]-selected_lidars[:,1])
            #                                
                                                nearest_lidar = np.argmin(lidar_dist)
                                            
                                             
                                                selected_height = heights_list[nearest_lidar]
                                             
                                            
                                                heights[:,k]=selected_height
                                        else:
                                            continue
                                      
        #                                heights = np.zeros((1,points.shape[1]))
        #                                selected_heights=[]
        #                                for k in range(points.shape[1]):
        #                                    lidar_dist = np.square(points[0,k]-bev_lidar_points[0,:])+np.square(points[1,k]-bev_lidar_points[1,:])
        ##                                
        #                                    nearest_lidar = np.argmin(lidar_dist)
        #                                    
        #                                    selected_lidar = bev_lidar_points[:,nearest_lidar]
        #                                    if selected_lidar[2] < lidar_height_thresh:
        #                                 
        #                                        selected_heights.append(selected_lidar[2])  
        #                                 
        #                                    
        #                                        heights[:,k]=selected_lidar[2]
        #                                    
        #                                    
        ##                                        heights = np.ones((1,points.shape[1]))*np.mean(heights_list)
        #                                if len(selected_heights) < 1:
        #                                    continue
                                        
                                    else:
                                        heights = np.zeros((1, points.shape[1]))
                                    
        #                                points = np.vstack((points, np.zeros((1, points.shape[1]))))
                                    points = np.vstack((points, heights))
                                    
        #                            points = np.vstack((points, np.zeros((1, points.shape[1]))))
                
                                    # Transform into the ego vehicle frame for the timestamp of the image.
                                    points = points - np.array(poserecord['translation']).reshape((-1, 1))
                                    points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)
                
                                    # Transform into the camera.
                                    points = points - np.array(cs_record['translation']).reshape((-1, 1))
                                    points = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, points)
                                    
                                    camera_points = np.copy(points)
                                    # Remove points that are partially behind the camera.
                                    depths = points[2, :]
                #                    behind = ((depths < near_plane) | (depths > 25*np.sqrt(2)))
                                    behind = depths < near_plane
                                    if np.all(behind):
                                        continue
                
                                    if render_behind_cam:
                                        # Perform clipping on polygons that are partially behind the camera.
                                        points,_ = _clip_points_behind_camera(points, near_plane)
                                    elif np.any(behind):
                                        # Otherwise ignore any polygon that is partially behind the camera.
                                        continue
                
                                    # Ignore polygons with less than 3 points after clipping.
                                    if len(points) == 0 or points.shape[1] < 3:
                                        continue
                
                                    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                                    points, norm_const = view_points(points, cam_intrinsic, normalize=True)
                
                                    # Skip polygons where all points are outside the image.
                                    # Leave a margin of 1 pixel for aesthetic reasons.
                                    inside = np.ones(points.shape[1], dtype=bool)
                                    inside = np.logical_and(inside, points[0, :] > 1)
                                    inside = np.logical_and(inside, points[0, :] < im.size[0] - 1)
                                    inside = np.logical_and(inside, points[1, :] > 1)
                                    inside = np.logical_and(inside, points[1, :] < im.size[1] - 1)
                                    if render_outside_im:
                                        if np.all(np.logical_not(inside)):
                                            continue
                                    else:
                                        if np.any(np.logical_not(inside)):
                                            continue
                
                                    points = points[:2, :]
                                    points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
                                    polygon_proj = Polygon(points)
                                        
                                    
        #                            polygon_proj = polygon_proj.intersection(non_occ_polygon_proj)
                                    # Filter small polygons
                                    if polygon_proj.area < min_polygon_area:
                                        continue
                                    
                                    layer_list.append(np.array(points))
                                label = layer_name
#                                ax.add_patch(descartes.PolygonPatch(polygon_proj, fc=color_map[layer_name], alpha=alpha,
#                                                                    label=label))
    
                        except Exception as e:
                            
                            logging.error(str(e))
                            continue
            vertices_per_layer[layer_name] = layer_list
                            
#                        break

        # Display the image.
#        plt.axis('off')
#        ax.invert_yaxis()
        
#        plt.tight_layout()
#        plt.savefig('/home/cany/mapmaker/no_alignment_narrow.png', bbox_inches='tight', pad_inches=0)
#        if out_path is not None:
#            plt.tight_layout()
#            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
            
        return np_img, vertices_per_layer,my_lidar_mask    
            
def load_point_cloud(nuscenes, pointsensor):

    # Load point cloud
    lidar_path = os.path.join(nuscenes.dataroot, pointsensor['filename'])
    pcl = LidarPointCloud.from_file(lidar_path)
    return pcl



def signedVolume(a, b, c, d):
    """Computes the signed volume of a series of tetrahedrons defined by the vertices in 
    a, b c and d. The ouput is an SxT array which gives the signed volume of the tetrahedron defined
    by the line segment 's' and two vertices of the triangle 't'."""

    return np.sum((a-d)*np.cross(b-d, c-d), axis=2)

def segmentsIntersectTriangles(s, t):
    """For each line segment in 's', this function computes whether it intersects any of the triangles
    given in 't'.
    
    s : 2xSx3
    t : 3XTX3
    """
    # compute the normals to each triangle
    normals = np.cross(t[2]-t[0], t[2]-t[1])
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # get sign of each segment endpoint, if the sign changes then we know this segment crosses the
    # plane which contains a triangle. If the value is zero the endpoint of the segment lies on the 
    # plane.
    # s[i][:, np.newaxis] - t[j] -> S x T x 3 array
    sign1 = np.sign(np.sum(normals*(s[0][:, np.newaxis] - t[2]), axis=2)) # S x T
    sign2 = np.sign(np.sum(normals*(s[1][:, np.newaxis] - t[2]), axis=2)) # S x T

    # determine segments which cross the plane of a triangle. 1 if the sign of the end points of s is 
    # different AND one of end points of s is not a vertex of t
    cross = (sign1 != sign2)*(sign1 != 0)*(sign2 != 0) # S x T 

    # get signed volumes
    v1 = np.sign(signedVolume(t[0], t[1], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T
    v2 = np.sign(signedVolume(t[1], t[2], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T
    v3 = np.sign(signedVolume(t[2], t[0], s[0][:, np.newaxis], s[1][:, np.newaxis])) # S x T

    same_volume = np.logical_and((v1 == v2), (v2 == v3)) # 1 if s and t have same sign in v1, v2 and v3

    return cross*same_volume > 0
#

def get_occlusion_mask(lidar_points,extents,resolution,lidar_height):
    
    orig_lidar_points = np.copy(lidar_points)
    
#    my_points = lidar_points[0::2]
    ground_points = lidar_points[1,:] > (lidar_height -0.3)
    
    lidar_points = lidar_points[:,ground_points]
    my_points_x = lidar_points[0]
    my_points_z = lidar_points[2]
    
    my_slopes = my_points_z/my_points_x
    
    x1, z1, x2, z2 = extents
    
    x = np.arange(x1-resolution, x2+resolution, resolution)
    z = np.arange(z1-resolution, z2+resolution, resolution)
    z = np.flip(z)
    
#    & (my_points_z <= z_max)
    my_mask = np.squeeze(np.zeros((lidar_points.shape[1],1)))
    
    
    for k in range(len(x)-2):
        cur_k = k+1
        for m in range(len(z)-2):
            cur_m = m+1
            x_min = (x[cur_k-1] + x[cur_k])/2
            x_max = (x[cur_k] + x[cur_k+1])/2
            z_max = (z[cur_m-1] + z[cur_m])/2
            z_min = (z[cur_m] + z[cur_m+1])/2
            
            if x_max < 0:
                min_slope = z_max/x_max
                max_slope = z_min/x_min
            else:
            
                min_slope = z_min/x_max
                max_slope = z_max/x_min
            
#            if ((min_slope < 0) & (max_slope < 0)):
#                temp = np.copy(min_slope)
#                min_slope=np.copy(max_slope)
#                max_slope = temp
#            
#            elif ((min_slope > 0) & (max_slope < 0)):
#                
#                temp = np.copy(min_slope)
#                min_slope=np.copy(max_slope)
#                max_slope = temp
            
            
            
            slope_cond = (my_slopes <= max_slope) & (my_slopes >= min_slope)
#            logging.error('MIN SLOPE : ' + str(min_slope) + ' MAX SLOPE : ' + str(max_slope)+' COND : ' + str(np.any(slope_cond)))
            if np.any(slope_cond):
                
                
                
                selected_x = my_points_x[slope_cond]
                selected_z = my_points_z[slope_cond]
#                if np.any((my_points_x >= x_min) & (my_points_x <= x_max) & (my_points_z >= z_min)  & (my_points_z <= z_max)):
                if np.any((np.abs(selected_x) >= np.abs(x_min)) & (selected_z >= z_min) ):
                    my_mask[(np.abs(my_points_x) >= np.abs(x_min)) & (my_points_z >= z_min)& slope_cond] = 1
                
    return lidar_points[:,my_mask > 0.5]


def get_lidar_map(lidar_points,extents,resolution, lidar_height):
    
#    my_points = lidar_points[0::2]
    
    ground_points = lidar_points[1,:] > (lidar_height -1)
    
    lidar_points = lidar_points[:,ground_points]
    
    my_points_x = lidar_points[0]
    my_points_z = lidar_points[2]
    
    my_slopes = my_points_z/my_points_x
    
    x1, z1, x2, z2 = extents
    
    x = np.arange(x1-resolution, x2+resolution, resolution)
    z = np.arange(z1-resolution, z2+resolution, resolution)
    z = np.flip(z)
    
#    & (my_points_z <= z_max)
    my_mask = np.zeros((len(z)-2,len(x)-2))
    
    
    for k in range(len(x)-2):
        cur_k = k+1
        for m in range(len(z)-2):
            cur_m = m+1
            x_min = (x[cur_k-1] + x[cur_k])/2
            x_max = (x[cur_k] + x[cur_k+1])/2
            z_max = (z[cur_m-1] + z[cur_m])/2
            z_min = (z[cur_m] + z[cur_m+1])/2
            


            if np.any((my_points_x >= x_min) & (my_points_x <= x_max) & (my_points_z >= z_min)  & (my_points_z <= z_max)):
#                if np.any((np.abs(selected_x) >= np.abs(x_min)) & (selected_z >= z_min) ):
                my_mask[m,k] = 1
                
    return my_mask

def lidar_func(sample,nuscenes):
    
    nusc = nuscenes
    
    cam = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    pointsensor = nuscenes.get('sample_data', sample['data']['LIDAR_TOP'])
    pc = load_point_cloud(nuscenes, pointsensor)
    
    
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))
    
    bev_lidar = np.copy(pc.points)

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
    
    return np.copy(pc.points), bev_lidar
    
