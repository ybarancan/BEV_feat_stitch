import os
import sys
import numpy as np
from PIL import Image
import glob

import logging
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader \
    import ArgoverseTrackingLoader
# from argoverse.utils.camera_stats import RING_CAMERA_LIST


from dataset.utils import get_occlusion_mask, \
    encode_binary_labels
from dataset.argoverse_utils import get_object_masks, get_map_mask
from experiments import argoverse_objects_exp as exp_config

RING_CAMERA_LIST = [
    "ring_front_center"

]


def process_split(map_data, config):

    # Create an Argoverse loader instance
    path = exp_config.argo_track_path
    print("Loading Argoverse tracking data at " + path)
    loader = ArgoverseTrackingLoader(path)
    
    
    
    for scene in loader:
        # if scene.current_log in VAL_LOGS:
        process_scene( scene, map_data, config)


def process_scene( scene, map_data, config):

    logging.error("\n\n==> Processing scene: " + scene.current_log)

    # Iterate over each camera and each frame in the sequence
    for camera in RING_CAMERA_LIST:
        for frame in range(scene.num_lidar_frame):
            # progress.update(i)
            process_frame(scene, camera, frame, map_data, config)
            # i += 1
            

def process_frame( scene, camera, frame, map_data, config):

    # Compute object masks
    masks = get_object_masks(scene, camera, frame, config.map_extents,
                             config.map_resolution)
    
    # Compute drivable area mask
    masks[0] = get_map_mask(scene, camera, frame, map_data, config.map_extents,
                            config.map_resolution)
    
    # Ignore regions of the BEV which are outside the image
    calib = scene.get_calibration(camera)
    # masks[-1] |= ~get_visible_mask(calib.K, calib.camera_config.img_width,
    #                                config.map_extents, config.map_resolution)
    
    # Ignore regions of the BEV which are occluded (based on LiDAR data)
    lidar = scene.get_lidar(frame)
    cam_lidar = calib.project_ego_to_cam(lidar)
    masks[-1] = get_occlusion_mask(cam_lidar, config.map_extents, 
                                    config.map_resolution)
    
    # Encode masks as an integer bitmask
    labels = encode_binary_labels(masks)

    # Create a filename and directory
    timestamp = str(scene.image_timestamp_list_sync[camera][frame])
    output_path = os.path.join(exp_config.argo_labels_path, 
                               scene.current_log, camera, 
                               f'{camera}_{timestamp}.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save encoded label file to disk
    Image.fromarray(labels.astype(np.int32), mode='I').save(output_path)
    

if __name__ == '__main__':

    # Create an Argoverse map instance
    map_data = ArgoverseMap()

    process_split(map_data, exp_config)
