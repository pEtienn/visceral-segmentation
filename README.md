# visceral-segmentation
 Consolidated code for visceral segmenation

This document details the following processes:
1.Format the visceral dataset 
2.Train the dense vnet network on niftynet
3.Generate inferences with niftynet
4.Evaluate the inferences
5.Modify the inferences to allow generation of a Finite Element Model

All the functions mentionned in this document are included in this repository.
**********************************************************
1.Formatting the Visceral Dataset:
The visceral dataset contains the different segmentations in differents files for the same volume image. The 
first step is to combine the segmentations we are interested in. Use the fonction *CombineSegmentations* to do so. Some scans will be rejected at this point because they can't be read. 

After this perform the following steps:
-Select only the patients with the labels needed
-Normatlize the pixel dimensions on the label images and the volumes images
-Align and crop the images as much as possible
-Standardize the labels (from 0 to the number of labels)
-Produce images of the same size for each patient. (120,120,120) was used in the previous experiment.

Note on resizing images
Volume images: ndimage.zoom is used. It uses spline interpolation.
Label images: a custom function is used. It uses nearest neighbour interpolation.

Note on cropping and aligning
The same crop size is used for each image. The crop size is the smallest window that contain all labels in all images. "Aligning" is done taking by centering the crop window on the center of each image's label. A better approach would be using keypoints to align the images. 

All those steps are done automatically with the function *PrepareData*. But before using it the data needs to be manually filtered. First the data (result of *CombineSegmentations*) must be separated manually between full body images and torso images. This is so that the Align and Crop step succeeds. Then some data needs to be removed. Data with abnormal pixel dimension, inversed axis or generally properties that are far from their group will result in *PrepareData* failling. Use the function *GetVolumesInfo* to get the size and pixel dimensions of the images.

You should separate the data in 2 folders named (suggestion) fullBody and torso. Then the call of PrepareData is this: 
labelPath=r'path/fullBody'
mriPath=r'path/mri'
outPath=r'path/outpath'
PrepareData(labelPath,mriPath,outPath)
labelPath=r'path/torso'
PrepareData(labelPath,mriPath,outPath)

This will generate images that are ready to be inputed in the dense vnet. Those images have already been generated and are on Etienne's google drive named: vnet_inputs(contain_no_errors).zip

**********************************************************
2.train the dense vnet
Use the following command to train the a net with niftynet:
net_segment train -c ~/niftynet/extensions/dense_vnet_abdominal/config.ini

The config.ini file is available on Etienne's drive. This assumes a standard file configuration. To train the net you need to place the training data on the path specified in the config.ini file. This file also point to the output location. 

**********************************************************
3.Generate inferences with niftynet
Use the following command to infer segmentations:
net_segment inference -c ~/niftynet/extensions/dense_vnet_abdominal/config.ini

**********************************************************
4.Evaluate the inferences
To measure the DC of the inferences use the function *MesureDCFile*. You'll need the path of the output of niftynet as well as the path pointing to the true segmentations.

**********************************************************
5.Modify the inferences to allow generation of a Finite Element Model

Use the function UpsampleFiles. It:
    1.Normalize pixel dimension
    2.Upsample normalized image
	
The resulting image have a pixel dimension of (0.5,0.5,0.5)
	
The upsampling is made in a way that pixels of different class will never be adjacent (using 26-adjacency). Effectively zeros are inserted between adjacent classes.