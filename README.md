Official code for Understanding Bird’s-Eye View of RoadSemantics using an Onboard Camera - RAL/ICRA 2022

[Link to paper](https://arxiv.org/pdf/2012.03040.pdf) 


Written with Tensorflow.

Make sure you have installed Nuscenes and/or Argoverse devkits and datasets installed

## Data preparation

For Nuscenes, run make_nuscenes_labels.py and dataset_creator.py, by setting the necessary paths in the experiments/nuscenes_objects_base.py file.
For Argoverse, run make_argoverse_labels.py, by setting the necessary paths in the experiments/argoverse_objects_exp.py file.

## Training

For training, run the relavant dataset's train.py file. 

## Trained Models

We provide trained model for Nuscenes dataset:

https://data.vision.ee.ethz.ch/cany/BEV-stitch/bev-stitch-nusc.zip

