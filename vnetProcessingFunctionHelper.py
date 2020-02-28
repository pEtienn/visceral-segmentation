# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 20:51:52 2019

@author: Etenne Pepyn
"""
import os
from pathlib import Path
import numpy as np 
import nibabel as nib
import re
from numba import guvectorize,float64,int64
import shutil
import csv
from scipy import ndimage as ndi
import sys
import subprocess
import pandas


"""
This file contains a number of functions usefull in manipulating data for use in niftynet
main functions:
    PrepareDataForVnetInput : takes care of the whole process 
                    There's few checks on data, and the visceral datasets has some anomalies so data need to be selected manually in part
    UsampleFiles: use ready images for FEM
"""
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
#UTILITIES######################
def GetVolumesInfo(path):
    """
    Prints shape and pixel dimension for all images in path

    """
    srcPath=str(Path(path))
    allF=os.listdir(srcPath) 
    for f in allF:
        if f[-3:]=='nii' or f[-6:]=='nii.gz':
            img=nib.load(os.path.join(srcPath,f))
            arr=np.squeeze(img.get_fdata())
            h=img.header
            print(f[0:20],'\t','shape: ',arr.shape,'\t pixDim: ',h.get_zooms())
            
def getDC(computed,truth,value):
    """
    Mesure dice coefficient with numpy arrays

    """
    mapC=computed==value
    mapT=truth==value
    num=2*np.sum(np.logical_and(mapC,mapT))
    den=np.sum(mapC)+np.sum(mapT)
    return num/den

def MesureDCFile(truthPath,predictedPath):
    """
    Prints DC of each file outputed by niftynet
    truthPath: true segmentations of images infered by niftynet
    predictedPath: output of niftynet

    """
    rLabelNumber=re.compile('\d\d\d\d')
    truthPath=str(Path(truthPath))
    predictedPath=str(Path(predictedPath))
    allF=os.listdir(predictedPath)
    allSrc=os.listdir(truthPath)
    dc=[]
    for i in range(len(allF)):
        f=allF[i]
        if 'nii.gz' in f:
            number=rLabelNumber.findall(f)[0]
            imgPredicted=nib.load(os.path.join(predictedPath,f))
            arrPredicted=np.squeeze(imgPredicted.get_fdata())
            pathSource=os.path.join(truthPath,[x for x in allSrc if ((number in x) and ('Label' in x))][0])
            imgTruth=nib.load(pathSource)
            arrTruth=np.round(np.squeeze(imgTruth.get_fdata()))
            for j in range(np.unique(arrTruth).shape[0]):
                t=getDC(arrPredicted,arrTruth,j)
                print('DC ',j,' : ',t)
                dc.append(t)
            print('\n')
    return dc
######## KEY FILES FUNCTIONS ####################################
    
class keyInformationPosition():
    """
    information locations in SIFT datapoints
    in an array of SIFT keypoints, row are the number of the keypoints, and colums
    contains it's characteristics. Use kIP object to access the right characteristics
    """
    scale=3
    XYZ=slice(0,3,1)
    descriptor=slice(17,81,1)
    flag=16
    noFlag=[x for x in range(81) if x != 16]
    
kIP=keyInformationPosition()

def FilterKeysWithMask(k,mask,considerScale=False):
    k2=np.zeros(k.shape)
    invMask=~mask
    prevXYZ=np.array([1,2,3])
    transfered=0
    for i in range(k.shape[0]):
        XYZ=np.int32(k[i,kIP.XYZ])
        if ~np.allclose(XYZ,prevXYZ): #used to skip keypoints with different rotation
            if considerScale==False:
                if mask[tuple(XYZ)]==True:
                    k2[i,:]=k[i,:]
                    transfered=1
                else:
                    transfered=0
            else:
                mask2=np.zeros(mask.shape,dtype=np.bool)
                c=int(k[i,kIP.scale]/np.sqrt(3)) #half the side of the smallest cube inside the keypoint
                mask2[XYZ[0]-c:XYZ[0]+c,XYZ[1]-c:XYZ[1]+c,XYZ[2]-c:XYZ[2]+c]=True
                if ~np.any(np.logical_and(invMask,mask2)):
                    k2[i,:]=k[i,:]         
                    transfered=1
                else:
                    transfered=0
        elif transfered==1:
            k2[i,:]=k[i,:]
        prevXYZ=XYZ
    k2=k2[~np.all(k2==0,axis=1)]
    return k2

def ReadKeypoints(filePath):
    filePath=str(Path(filePath))
    file= open (filePath,'r')
    i=0
    end=0
    while end==0:
        r=file.readline()
        if 'Scale-space' in r:
            end=1
        i=i+1
    file.close()
    a=np.arange(0,i)
    listC=list(a)
    df=pandas.read_csv(filePath,sep='\t',header=None,index_col=False,skiprows=listC)
    if len(df.columns)==82: #happens when there are tab at the end of a line (shouldnt be there)
        df=df.drop(81,axis=1)

    key=df.to_numpy(dtype=np.float64)

    return key

def GetResolutionHeaderFromKeyFile(filePath):
    rReso=re.compile('(\d+ \d+ \d+)')
    
    file= open (filePath,'r')
    #skip
    header=''
    end=0
    while end==0:
        r=file.readline()
        header=header+r
        if 'resolution' in r or 'Resolution' in r:
            resoString=rReso.findall(r)
            if resoString ==[]:
                print('resolution not found in'+filePath)
            else:
                resoString=resoString[0]
            resolution=[int(i) for i in resoString.split()]
        if 'Scale-space' in r:
            end=1
    if r[:5]!='Scale':
        print('ERROR: keypoint format not supported in'+filePath)
    file.close()
    if not('resolution' in locals()):
        resolution=np.array([0,0,0])
    return [resolution,header]


def WriteKeyFile(path,keys,header='default'):
    #todo change the number of features on given headers
    if header=='default':
        header="# featExtract 1.1\n# Extraction Voxel Resolution (ijk) : 176 208 176\
        \n#Extraction Voxel Size (mm)  (ijk) : 1.000000 1.000000 1.000000\
        \n#Feature Coordinate Space: millimeters (gto_xyz)\nFeatures: " +str(keys.shape[0])+"\
        \nScale-space location[x y z scale] orientation[o11 o12 o13 o21 o22 o23 o31 o32 o32] 2nd moment eigenvalues[e1 e2 e3] info flag[i1] descriptor[d1 .. d64]\n"
    else:
        p=re.compile('Features: \d+')
        header=p.sub('Features: '+str(keys.shape[0])+' ',header)
    fW=open(path,'w',newline='\n')
    fW.write(header)
    for i in range(keys.shape[0]):
        for j in range(keys.shape[1]):
            if j>=16:
                n=str(int(keys[i,j]))
                fW.write(n+'\t')
            else:
                n=keys[i,j]
                fW.write(format(n,'.6f')+'\t')
            
        fW.write('\n')
    fW.close()
    
##########################################################################
def ReadImage(filePath):
    filePath=str(Path(filePath))
    img=nib.load(filePath)
    arr=np.squeeze(img.get_fdata())
    h=img.header
    return[arr,h]    
    
def SaveImage(filePath,arr,header):
    im2=nib.Nifti1Image(arr,affine=None,header=header)
    nib.save(im2,filePath)
    
def CombineSegmentations(srcPath,outPath):
    """
    Takes segmentation files of the Visceral dataset and creates a new set of files.
    There's one new file per patient, containing all segmentations.
    Funciton skips files causing problems.
     *** INPUT ***
    srcPath: path of the segmentation folder
    """

    rLabelNumber=re.compile('(wb|Ab)_([0-9]+)_')
    rVolumeID=re.compile('(^.+[wb|Ab])_')
    srcPath=str(Path(srcPath))
    outPath=str(Path(outPath))
    allF=os.listdir(srcPath)
    unreadableFile=' '
    previousOutput=' '
    affine = np.diag([1, 2, 3, 1])

    for f in allF:  
        shortName=rVolumeID.findall(f)[0]

        if shortName!=unreadableFile:
            outputName=(os.path.join(outPath,shortName+'.nii.gz'))
            try:
                img=nib.load(os.path.join(srcPath,f))
                arr=img.get_fdata()

            except:
                print('problem loading '+f+', skipping '+shortName)
                unreadableFile=shortName

                
            if shortName!=unreadableFile:
                if previousOutput!=outputName:

                    h=img.header
                    shape=h.get_data_shape()   
                    if previousOutput!=' ' and previousOutput!= os.path.join(outPath,unreadableFile+'.nii.gz'):

                        print(previousOutput)
                        segArray=np.int32(segArray)
                        arrayImg = nib.Nifti1Image(segArray,affine=None,header=h)

                        nib.save(arrayImg,previousOutput) 
                    segArray=np.zeros(shape)
                
                uni=np.unique(arr)
                if np.shape(uni)[0]==2:
                    k=rLabelNumber.findall(f)[0][1]
                    n=int(k)
                    if uni[1]!=n:  
                        arr=(arr>0)*n
                else:
                    arr=0
                segArray=segArray+arr*(segArray==0)
        previousOutput=outputName
    
    if shortName!=unreadableFile:
        segArray=np.int32(segArray)
        arrayImg = nib.Nifti1Image(segArray, affine=None,header=h)
        nib.save(arrayImg,outputName) 
        
    print(previousOutput)

def SelectLabelInSegmentation(srcPath,outPath,labelToKeep):
    """
    Takes all the segmentation files generated by CombineSegmentations and generates new
    segmentations files containing only labels specified by labelToKeep
     *** INPUT ***
    srcPath: path of the output of CombineSegmentations
    outPath: destination of new segmentations files
    labelToKeep: list containing labels to keep
    """
    affine = np.diag([1, 2, 3, 1])
    allF=os.listdir(srcPath)
    for f in allF: 
        img=nib.load(os.path.join(srcPath,f))
        arr=img.get_fdata()
        h=img.header
        outArr=np.zeros(arr.shape)
        for i in range(len(labelToKeep)):
            outArr=outArr+(arr==labelToKeep[i])*labelToKeep[i]
        arrayImg = nib.Nifti1Image(outArr,affine=None,header=h)
        nib.save(arrayImg,os.path.join(outPath,f)) 
        
def GetLabelInPatient(srcPath,labelList):
    """
    Tool to check the presence of some labels in combined segmentation files, 
    or all the labels present if labelList=[]
     *** INPUT ***
    srcPath:folder containing combined segmentation files
    labelList: either list of labels you want to verify the presence,
            or empty, to show all the labels present in each file
     *** OUTPUT ***
    The present labels will be shown on the console after the name of the file
    allLabels: matrix n*2, 
            colums: labelNb | nb Of Patient With That Label     
    """
    np.set_printoptions(precision=3)
    allF=os.listdir(srcPath)
    allLabels=np.zeros((1,2),dtype=np.int64)

    for f in allF: 
        img=nib.load(os.path.join(srcPath,f))
        arr=img.get_fdata()
        print (f,end=" ")
        if not labelList :
            u=np.unique(arr)
            for l in u:
                print(l,end=" ")
                if np.sum(l==allLabels[:,0],0)==0:
                    allLabels=np.concatenate((allLabels,np.array([[int(l),1]])),axis=0)
                else:
                    allLabels[allLabels[:,0]==l,1]+=1
        else:
            for l in labelList:
                if np.sum(arr==l)!=0:
                    print(l,end=" ")
                    if np.sum(l==allLabels[:,0],0)==0:
                        allLabels=np.concatenate((allLabels,np.array([[int(l),1]])),axis=0)
                    else:
                        allLabels[allLabels[:,0]==l,1]+=1
        print(' ')
        if np.sum(allLabels==256)>0:
            pass
    return allLabels[allLabels[:,0].argsort()]

def FilterPatientsByLabel(srcPath,outPath,labelList):
    """
    Keep only patients with the required labels and throw away all unrequired labels

    """
    outPath=str(Path(outPath))
    srcPath=str(Path(srcPath))
    allF=os.listdir(srcPath)
    
    for f in allF:
        img=nib.load(os.path.join(srcPath,f))
        arr=np.squeeze(img.get_fdata())
        arr2=np.zeros(arr.shape)
        allPresent=1
        for i in labelList:
            if np.sum(arr==i)>0:
                arr2[arr==i]=arr[arr==i]
            else:
                allPresent=0
                break
        if allPresent==1:
            im2=nib.Nifti1Image(arr2,affine=None,header=img.header)
            nib.save(im2,os.path.join(outPath,f))


def GetCropSize(refLabelPath,refOriginalKeyPath,refKeyPath):
    """
    Determine :window size to keep on ref image, scaling to obtain desired window size
    Filter reference keypoints present in desired segmentations
    
    Parameters
    ----------
    path : 

    Returns
    -------
    [radius XYZ of the window, center XYZ of the window]

    """
    marginRatio=0.05
    imageBoxCenter=np.zeros(3)

    img=nib.load(refLabelPath)
    arr=np.float32(np.squeeze(img.get_fdata()))
    a=np.nonzero(arr)
    XYZ=np.zeros((3,2),dtype=np.int64)
    for i in range(3):
        XYZ[i,0]=np.min(a[i])
        XYZ[i,1]=np.max(a[i])+1
#        print(XYZ)
    imageBoxCenter[:]=np.mean(XYZ,axis=1)    

    XYZ=np.ceil(XYZ)
    XYZ=(XYZ+XYZ%2)/2
    sizeXYZ=XYZ[:,1]-XYZ[:,0]
    #add margin
    margin=(marginRatio*sizeXYZ)
    margin=np.int64(margin)
    XYZ[:,0]+=-margin
    XYZ[:,1]+=margin
    sizeXYZ=XYZ[:,1]-XYZ[:,0]
    windowSize=sizeXYZ
    
    #generate keyRef
    k=ReadKeypoints(refOriginalKeyPath)
    [reso,h]=GetResolutionHeaderFromKeyFile(refOriginalKeyPath)
    mask=arr>0
    k1=FilterKeysWithMask(k,mask)
    WriteKeyFile(refKeyPath,k1,h)
    
    return [windowSize,imageBoxCenter]

def Crop(srcPath,outPath,XYZ,imageBoxCenter,margin=[0,0,0],decompress=True):
    """
    Crop all images in path
    XYZ: output from GetCropBoundaries
    Margin: [x,y,z] margin to add to the boudary boxes, the margin will be added 2 times on each axis
    *** OUTPUT ***
     modify directly the images
     """
    margin=np.int64(margin)
    XYZ=np.int64(XYZ)
    imageBoxCenter=np.int64(imageBoxCenter)
    outPath=str(Path(outPath))
    srcPath=str(Path(srcPath))
    allF=os.listdir(srcPath) 
    for i in range(len(allF)):
        f=allF[i]
        img=nib.load(os.path.join(srcPath,f))
        arr=np.int16(np.squeeze(img.get_fdata()))
        nh=img.header
        cubeLimit=np.zeros((3,2))
        for j in range(3):
                cubeLimit[j,0]=max(imageBoxCenter[i,j]-XYZ[j]-margin[j],0)
                cubeLimit[j,1]=min(imageBoxCenter[i,j]+XYZ[j]+margin[j]-1,arr.shape[j]-1)
        cubeLimit=np.int64(cubeLimit)
        newArr=arr[cubeLimit[0,0]:cubeLimit[0,1],cubeLimit[1,0]:cubeLimit[1,1],cubeLimit[2,0]:cubeLimit[2,1]]
        imgv2=nib.Nifti1Image(newArr,affine=None,header=nh)
        if decompress==True:
            nib.save(imgv2,os.path.join(outPath,f[:-3]))
        else:
            nib.save(imgv2,os.path.join(outPath,f))

def ExtractAllKeys(volumeNormalizedPath,dstPath,volumeID='(\d{4})[_.]'):
    rVolumeID=re.compile(volumeID)
    allF=os.listdir(volumeNormalizedPath)
    print('extracting keypoints')
    for f in allF:
        num=rVolumeID.findall(f)[0]
        print(num)
        list_files = subprocess.run(['./featExtract.exe',os.path.join(volumeNormalizedPath,f),os.path.join(dstPath,num+'.key')])
        print("The exit code was: %d" % list_files.returncode)
        
def CalculateTransformationMatrices(pKeyRef,pKeyFolder,volumeID='(\d{4})[_.]'):
    rVolumeID=re.compile(volumeID)
    allF=os.listdir(pKeyFolder)
    print('calculating transformation matrices')
    transformList=[]
    for f in allF:
        
      #  if not rVolumeID.findall(f):#only match with non-ref patients
        #list_files = subprocess.run(['./featMatchMultiple.exe','-r-',pKeyRef,os.path.join(pKeyFolder,f)]) #,'-n','1'
        command=[r'./featMatchMultiple.exe',r'-r-',pKeyRef,os.path.join(pKeyFolder,f)] #
        output = subprocess.run(command,stdout=subprocess.PIPE)
        inlierReg='inliers (\d+)'
        rInlier=re.compile(inlierReg)

        # print(str(output.stdout))
        nbInliers=rInlier.findall(str(output.stdout))[0]
        print('nb of inliers: ',nbInliers)
        #print("The exit code was: %d" % list_files.returncode)
        pInvTrans=os.path.join(pKeyFolder,f+'.trans-inverse.txt') #
        df=pandas.read_csv(pInvTrans,sep='\t',header=None)
        invTrans=df.to_numpy(dtype=np.float64)
        print(invTrans)
        transformList.append(invTrans)
    
    #delete non-key files
    newAllF=os.listdir(pKeyFolder)
    diff=[x for x in newAllF if x not in allF]
    for f in diff:
        os.remove(os.path.join(pKeyFolder,f))
    return transformList
    
    
def TransformImages(pRefKey,pImageFolder,pDstFolder,lTransform):
    allF=os.listdir(pImageFolder)
    if len(allF)!=len(lTransform):
        print('in TransformImages nb of files and nb of transforms did not match')
        sys.exit()
    [r,h]=GetResolutionHeaderFromKeyFile(pRefKey)
    for i in range(len(allF)):
        f=allF[i]
        [arr,h]=ReadImage(os.path.join(pImageFolder,f))
        newArr=ndi.affine_transform(arr,lTransform[i],output_shape=tuple(r))
        SaveImage(os.path.join(pDstFolder,f),newArr,h)
    
def Rename(srcPath,suffix,extension='.nii.gz',numberDetectionRegex='^[0-9]*([0-9]{4})_'):
    """
    Rename all files in srcPath

    """
    rPatientNumber=re.compile(numberDetectionRegex)
    srcPath=str(Path(srcPath))
    allF=os.listdir(srcPath) 
    for f in allF:
        num=rPatientNumber.findall(f)[0]
        os.rename(os.path.join(srcPath,f),os.path.join(srcPath,num+suffix+extension))
        
def GetCorrespondingVolumes(volumesPath,outPath,labelPath,volumeID='^(.+)[wb|Ab]'):
    """
    Selects volumes in volumesPath that are from the same patients as those
    in labelPath and puts them in outPath
    """
    rVolumeID=re.compile(volumeID)
    volumesPath=str(Path(volumesPath))
    outPath=str(Path(outPath))
    labelPath=str(Path(labelPath))
    
    allHdrF=os.listdir(volumesPath)
    allLabelF=os.listdir(labelPath) #we take only hdr files that also have a label
    for f in allLabelF:
        num=rVolumeID.findall(f)[0]
        hdrF=[x for x in allHdrF if (num in x)][0]
        img=nib.load(os.path.join(volumesPath,hdrF))
        nib.save(img,os.path.join(outPath,hdrF))
        
def StandardizeLabels(srcPath,outPath,labels):
    """
    Change the labels to [0,1,2,...,n] n being the number of labels

    """
    srcPath=str(Path(srcPath))
    allF=os.listdir(srcPath) 
    for f in allF:
        img=nib.load(os.path.join(srcPath,f))
        arr=np.int16(np.squeeze(img.get_fdata()))
        nh=img.header
        for i in range(len(labels)):
            arr[arr==labels[i]]=i
        imgv2=nib.Nifti1Image(arr,affine=None,header=nh)
        nib.save(imgv2,os.path.join(outPath,f))
                    
                    
@guvectorize([(int64[:,:,:],int64[:,:,:],int64[:,:,:])], '(x,y,z),(a,b,c)->(a,b,c)',nopython=True)
def ImResize3D(im,dum,im2):
    """
     resizes 3D image using NN. To be used on label images
     *** INPUT ***
    im: image
    dum: dummy array to have dimensions for the output array
    im2: output array
     *** OUTPUT ***
    new resized Image
    """
    newDim=np.asarray(dum.shape)
    oldDim=np.asarray(im.shape)
    for x in range(newDim[0]):
        for y in range(newDim[1]):
            for z in range(newDim[2]):
                im2[x,y,z]=im[np.int64(np.round_((oldDim[0]-1)/(newDim[0]-1)*x)),np.int64(np.round_((oldDim[1]-1)/(newDim[1]-1)*y)),np.int64(np.round_((oldDim[2]-1)/(newDim[2]-1)*z))]
           

def NormalizePixDimensions(folderPath,outPath):
    """
    Resize the window so that pixel dimension is (1,1,1)
    To use on label images.
    """
    path=str(Path(folderPath))
    outPath=str(Path(outPath))
    allF=os.listdir(path) 
    for f in allF:
        img=nib.load(os.path.join(path,f))
        arr=np.squeeze(img.get_fdata())
        h=img.header
        pixDim=np.asarray(h.get_zooms())
        oldDim=np.asarray(arr.shape)
        newDim=np.int64(np.multiply(np.float32(oldDim),pixDim))
        newArr=np.zeros(newDim,dtype=np.int64)
        dum=np.copy(newArr)
        ImResize3D(np.int64(arr),dum,newArr)
        h.set_zooms((1,1,1))
        imgv2=nib.Nifti1Image(newArr,affine=None,header=h)
        nib.save(imgv2,os.path.join(outPath,f))
        print(f)

        
def Zoom(img,newDimensions=[],newPixDim=[]):
    """
    Wrapper for ndimage.zoom to work on files

    """
    arr=np.squeeze(img.get_fdata())
    h=img.header
    pixDim=np.asarray(h.get_zooms()[0:3])
    oldDim=np.asarray(arr.shape)
    if np.any(newDimensions):
        zoomValue=newDimensions/oldDim
        newPixDim=pixDim/zoomValue
    else:
        zoomValue=pixDim/newPixDim
    arr2=ndi.zoom(arr,zoomValue)
    
    h.set_zooms(newPixDim)
    im2=nib.Nifti1Image(arr2,affine=None,header=h)
    return im2

def ZoomAllIm(srcPath,dstPath,newDimensions=[],newPixDim=[]):
    """
    Applyl Zoom to all volume images in srcPath

    """
    srcPath=str(Path(srcPath))
    dstPath=str(Path(dstPath))
    allF=os.listdir(srcPath)
    for i in range(len(allF)):
        f=allF[i]
        if 'MRI' in f:
            img=nib.load(os.path.join(srcPath,f))
            im2=Zoom(img,newDimensions,newPixDim)
            nib.save(im2,os.path.join(dstPath,f))
            print(f)

def ZoomAllLabel(srcPath,dstPath,newDimensions):
    """
    Apply ImResize3D to all label images in srcPath
    """
    srcPath=str(Path(srcPath))
    dstPath=str(Path(dstPath))
    allF=os.listdir(srcPath)
    for i in range(len(allF)):
        f=allF[i]
        if 'Label' in f:
            img=nib.load(os.path.join(srcPath,f))
            arr=np.squeeze(img.get_fdata())
            h=img.header
            pixDim=np.asarray(h.get_zooms())
            oldDim=np.asarray(arr.shape)
            zoomValue=newDimensions/oldDim
            newArr=np.zeros(newDimensions,dtype=np.int64)
            dum=np.copy(newArr)
            ImResize3D(np.int64(arr),dum,newArr)
            newPixDim=pixDim/zoomValue
            h.set_zooms(newPixDim)
            im2=nib.Nifti1Image(newArr,affine=None,header=h)
            nib.save(im2,os.path.join(dstPath,f))
            print(f)

       
def PrepareDataForVnetInput(labelPath,mriPath,outPath,labelList=[0,58,86,237,29193,29662,29663,32248,32249],imSize=np.array([120,120,120])):
    """
     prepare data for the dense vnet
     *** INPUT ***
    labelPath:path containing result of CombineSegmentation
    mriPath: path containing needed volumes (or more)
    outputPath: path to output the prepared data
    labelList: labels that image in the database must contain
     *** OUTPUT ***
    Images in the outputPath ready for the denseVnet
    normalized pixel dimension
    All cropped at the same window size
    Renamed
    Labels are go from 0 to n
    
    """
    labelPath=str(Path(labelPath))
    mriPath=str(Path(mriPath))
    outPath=str(Path(outPath))
    tFN=10
    tempPathList=[]
    for i in range(tFN):
        tempPathList.append(os.path.join(outPath,'tp'+str(i)))
        os.makedirs(tempPathList[i])
    FilterPatientsByLabel(labelPath,tempPathList[0],labelList)
    GetCorrespondingVolumes(mriPath,tempPathList[1],tempPathList[0])
    NormalizePixDimensions(tempPathList[0],tempPathList[2])
    NormalizePixDimensions(tempPathList[1],tempPathList[3])
    
    [XYZ,centers]=GetCropSize(tempPathList[2])
    Crop(tempPathList[2],tempPathList[4],XYZ,centers)
    Rename(tempPathList[3],'_MRI',extension='.nii.gz')
    Crop(tempPathList[3],tempPathList[6],XYZ,centers)
    Rename(tempPathList[4],'_Label',extension='.nii')
    StandardizeLabels(tempPathList[4],tempPathList[5],labelList)
    ZoomAllIm(tempPathList[6],outPath,newDimensions=imSize)
    ZoomAllLabel(tempPathList[5], outPath,imSize)
    for i in range(tFN):
        shutil.rmtree(tempPathList[i])
    
    
    return XYZ


            
@guvectorize([(int64[:,:,:],int64[:,:,:],int64[:,:,:])], '(x,y,z),(a,b,c)->(a,b,c)',nopython=True)
def _Upsample(im,dummy,imOutPadded):
    """
    Upsample an image to ready it for FEA
    """
    [row,col,deep]=np.asarray(np.shape(im))
    [row2,col2,deep2]=np.asarray(np.shape(dummy))
    for i in range(1,row2,2):
            for j in range(1,col2,2):
                for k in range(1,deep2,2):
                    imOutPadded[i,j,k]=im[np.int64(i/2),np.int64(j/2),np.int64(k/2)]
            
    for i in range(1,int(row*2)):
        for j in range(1,int(col*2)):
            for k in range(1,int(deep*2)):
                if imOutPadded[i,j,k]==0:
                    cube=imOutPadded[i-1:i+2,j-1:j+2,k-1:k+2]
                    u=np.unique(cube)
                    if u.shape[0]==2:
                       dummy[i,j,k]= u[1]
    imOutPadded+=dummy
    
                        
def Upsample(im):
    """
    Wrapper for _Upsample
    Upsample an image to ready it for FEA

    Parameters
    ----------
    im : Image as numpy array int64
    Returns
    -------
    im2: Image as numpy array

    """
    if im.dtype!='int64':
        sys.exit('input need to be an np.int64')
    s=np.asarray(im.shape)
    s=(s*2)+1
    imPadded=np.zeros(s,dtype=np.int64)
    dummy=np.copy(imPadded)
    _Upsample(im,dummy,imPadded)
    # imPadded=__Upsample(im,dummy,imPadded)
    im2=imPadded[1:-1,1:-1,1:-1]

    return im2

def UpsampleFiles(srcPath,dstPath):
    
    """
    srcPath: path of the output of niftynet
    dstPath: path to put the upsampled images in
    
    Upsample process to apply on the output of niftynet:
    1.Normalize pixel dimension
    2.Upsample normalized image
    3.Resulting image has a pixel dimension of (0.5,0.5,0.5)

    """
    srcPath=str(Path(srcPath))
    dstPath=str(Path(dstPath))
    allF=os.listdir(srcPath)
    for i in range(len(allF)):
        f=allF[i]
        if 'nii.gz' in f:
            img=nib.load(os.path.join(srcPath,f))
            arr=np.squeeze(img.get_fdata())
            arr=np.int64(arr)
            h=img.header
            
            #normalizing
            pixDim=np.asarray(h.get_zooms()[0:3])
            oldDim=np.asarray(arr.shape)
            newDim=np.int64(np.multiply(np.float32(oldDim),pixDim))
            newArr=np.zeros(newDim)
            dum=np.copy(newArr)
            ImResize3D(arr,dum,newArr)
        
            
            oldDim=np.asarray(newArr.shape)
            newArr2=Upsample(np.int64(newArr))
            newDimensions=np.asarray(newArr2.shape)
            zoomValue=newDimensions/oldDim           
            newPixDim=pixDim/zoomValue
            newPixDimOut=np.ones(5)
            newPixDimOut[0:3]=newPixDim
            h.set_zooms([0.5,0.5,0.5,1,1])
            im2=nib.Nifti1Image(newArr2,affine=None,header=h)
            nib.save(im2,os.path.join(dstPath,f))
            print(f)