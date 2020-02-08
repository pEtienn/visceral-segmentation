# visceral-segmentation
 Consolidated code for visceral segmenation

This document details the following processes:
1. Format the visceral dataset 
2. Train the dense vnet network on niftynet
3. Generate inferences with niftynet
4. Evaluate the inferences
5. Modify the inferences to allow generation of a Finite Element Model

All the functions mentionned in this document are included in this repository.
**********************************************************
1.Formatting the Visceral Dataset:

The following procedure will assume that your dataset is in a folder named *Anatomy3-trainingset* situated in the path *./Anatomy3-traininset*. This folder contains the folders *Landmarks*, *Segmentations* and *Volumes*. From now on, all paths will start from *./Anatomy3-traininset*. For example: */foo* is *./Anatomy3-traininset/foo*.

The visceral dataset contains segmentations in differents files for the same volume image, all in */Segmentations*. The first step is to separate the MRt1 segmentations that we will be using from the rest. To do so you can use the search fonction in window explorer with the keyword *MRT1* from the */Segmentations* folder, then copy and paste all files found in a folder named */mrt1*. The next step is to combine the segmentations in one file per patient. Use the function *CombineSegmentations* to do so, with the destination folder as */Combined_segmentations*. Some scans will be rejected at this point because they can't be read. 

Then perform the following steps with the function *PrepareData*:
* Select only the patients with the labels needed
* Normatlize the pixel dimensions on the label images and the volumes images
* Align and crop the images as much as possible
* Standardize the labels (from 0 to the number of labels)
* Produce images of the same size for each patient. (120,120,120) was used in the previous experiment.

All those steps are done automatically with the function *PrepareData*. But before using it, the data needs to be manually filtered. First, the content of */Combined_segmentations* must be separated manually between the folders */full_body* and */torso*. This is so that the Align and Crop step succeeds. Then some data needs to be removed. Data with abnormal pixel dimension, inversed axis or generally properties that are far from their group will result in *PrepareData* failling. Use the function *GetVolumesInfo* to get the size and pixel dimensions of the images.

After having separated the content of */Combined_segmentations* in */fullBody* and */torso* the call of PrepareData is this: 
<pre><code>
#in python r in front of a string signify it's a raw string, use it for paths
labelPath=r'/fullBody'
mriPath=r'/mri'
outPath=r'/outpath'
PrepareData(labelPath,mriPath,outPath)
labelPath=r'/torso'
PrepareData(labelPath,mriPath,outPath)
</code></pre>

This will generate images that are ready to be inputed in the dense vnet, both volumes and label images. Those images have already been generated and are named: *vnet_inputs(contain_no_errors).zip*

Note on resizing images
Volume images: *ndimage.zoom is used*. It uses spline interpolation.
Label images: a custom function is used. It uses nearest neighbour interpolation.

Note on cropping and aligning
The same crop size is used for each image. The crop size is the smallest window that contain all labels in all images. "Aligning" is done taking by centering the crop window on the center of each image's label. A better approach would be using keypoints to align the images. 

**********************************************************
2.train the dense vnet
Use the following command to train the a net with niftynet:
>net_segment train -c ~/niftynet/extensions/dense_vnet_abdominal/config.ini

This assumes a standard file configuration. To train the net you need to place the training data on the path specified in the config.ini file. This file also point to the output location. 

**********************************************************
3.Generate inferences with niftynet
Use the following command to infer segmentations:
>net_segment inference -c ~/niftynet/extensions/dense_vnet_abdominal/config.ini

**********************************************************
4.Evaluate the inferences
To measure the DC of the inferences use the function *MesureDCFile*. You'll need the path of the output of niftynet as well as the path pointing to the true segmentations.

**********************************************************
5.Modify the inferences to allow generation of a Finite Element Model

Use the function *UpsampleFiles*. It:
1. Normalize pixel dimension
2. Upsample normalized image
	
The resulting image have a pixel dimension of (0.5,0.5,0.5)
	
The upsampling is made in a way that pixels of different class will never be adjacent (using 26-adjacency). Effectively zeros are inserted between adjacent classes.


niftynet installation procedure

cuda 9.0 already installed

new conda env

python -m pip install --upgrade pip

install tensorflow-gpu==1.12 because of cuda version 9.0
https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781

https://www.tensorflow.org/install/source_windows#tested_build_configurations

