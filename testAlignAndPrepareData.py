# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:18:20 2020
This file contains the ongoing work to align images with a reference using keypoints.
The end-goal of this program is to process images before feeding them in a vnet
Right now there is too few keypoints per organs leading to irregular transforms
To use on you computer you will need to change the absolute paths below
    labelPath:path containing result of CombineSegmentation
    mriPath: path containing needed volumes (or more)
    outputPath: path to output the prepared data
    labelList: labels that image in the database must contain
    pKeyFolder: where the extracted keys will be stored.
    pTransformedImages: where transformed Images will be saved.
Right now the reference Image is image 21, it was determined abitrarily.
@author: Etienne Pepin
"""


from vnetProcessingFunctionHelper import *
import subprocess
import os
import re
import sys

labelPath=r'S:\Anatomy3-trainingset\combinedSegs_All'
mriPath=r'S:\Anatomy3-trainingset_Copy\Volumes'
outPath=r'C:\Users\Etenne Pepyn\niftynet\data\test_0225'
labelList=[0,58,86,237,29193,29662,29663,32248,32249]
imSize=np.array([120,120,120])

pKeyFolder=r'C:\Users\Etenne Pepyn\niftynet\data\test_0225\keys'
pTransformedImages=r'C:\Users\Etenne Pepyn\niftynet\data\test_0225\transformedImages'
refNum=21
refNumID='0('+str(refNum)+')[_.]'
rRefNumID=re.compile(refNumID)


tFN=10
tempPathList=[]
for i in range(tFN):
    tempPathList.append(os.path.join(outPath,'tp'+str(i)))
#     os.makedirs(tempPathList[i])
FilterPatientsByLabel(labelPath,tempPathList[0],labelList)
GetCorrespondingVolumes(mriPath,tempPathList[1],tempPathList[0])
NormalizePixDimensionsLabel(tempPathList[0],tempPathList[2]) #label
NormalizePixDimensionsVolume(tempPathList[1],tempPathList[3]) #volume

if not(os.path.isdir(pKeyFolder)):
    os.mkdir(pKeyFolder)
    ExtractAllKeys(tempPathList[3],pKeyFolder)

if not(os.path.isdir(pTransformedImages)):
    os.mkdir(pTransformedImages)
     
allLabelFile=os.listdir((tempPathList[2]))
allKeyFile=os.listdir(pKeyFolder)

refLabelList=[x for x in allLabelFile if rRefNumID.findall(x)]
refKeyList=[x for x in allKeyFile if rRefNumID.findall(x)]
if len(refLabelList)!=1 or len(refKeyList)!=1:
    print('Bad refNumber finding regex resulting in multiple or no reference found')
    print(refLabelList)
    print(refKeyList)
    sys.exit()
refLabelPath=os.path.join(tempPathList[2],refLabelList[0])
refOriginalKeyPath=os.path.join(pKeyFolder,refKeyList[0])
pKeyRef=r"C:\Users\Etenne Pepyn\niftynet\data\test_0225\ref\refKey.key"

[windowSize,imageBoxCenter]=GetCropSize(refLabelPath,refOriginalKeyPath,pKeyRef)
lInvTrans=CalculateTransformationMatrices(pKeyRef,pKeyFolder)
TransformImages(pKeyRef,tempPathList[3],pTransformedImages,lInvTrans)

# [XYZ,centers]=GetCropSize(tempPathList[2])
# Crop(tempPathList[2],tempPathList[4],XYZ,centers)
# Rename(tempPathList[3],'_MRI',extension='.nii.gz')
# Crop(tempPathList[3],tempPathList[6],XYZ,centers)
# Rename(tempPathList[4],'_Label',extension='.nii')
# StandardizeLabels(tempPathList[4],tempPathList[5],labelList)
# ZoomAllIm(tempPathList[6],outPath,newDimensions=imSize)
# ZoomAllLabel(tempPathList[5], outPath,imSize)
# for i in range(tFN):
#     shutil.rmtree(tempPathList[i])

