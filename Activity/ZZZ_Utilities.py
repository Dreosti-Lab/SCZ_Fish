# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:22:34 2022

@author: Tom
"""
import cv2
import numpy as np

def downSampleVid(downsampleFrames, aviFile, outName):
    try:
        print('Loading movie...')
        vid=cv2.VideoCapture(aviFile)
        print('...Done loading!')
        vid.set(cv2.CAP_PROP_POS_FRAMES, 99) # skip first 5 seconds (sometimes some bright artifact)
        ret,im=vid.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        width = np.size(im, 1)
        height = np.size(im, 0)
        vidOut= cv2.VideoWriter(outName,cv2.VideoWriter_fourcc(*'DIVX'), 10, (width,height),0)
        
        numFrames_orig = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        stepFrame=int(np.floor(np.divide(numFrames_orig,downsampleFrames)))
        print('Downsampling movie...')
        for idx,i in enumerate(np.arange(100,numFrames_orig, stepFrame)):
            print('Frame ' + str(idx) + ' of ' + str(len(np.arange(100,numFrames_orig, stepFrame))))
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, im = vid.read()
            current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            vidOut.write(current)
        print('...Done downsampling!')
        vidOut.release()
        vid.release()
        return 1
    except: 
        try:
            vidOut.release()
            vid.release()
            return 0
        except:
            return -1
    
def copyVid(aviFile, outName):
    try:
        print('Loading movie...')
        vid=cv2.VideoCapture(aviFile)
        print('...Done loading!')
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0) # skip first 5 seconds (sometimes some bright artifact)
        ret,im=vid.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        width = np.size(im, 1)
        height = np.size(im, 0)
        vidOut= cv2.VideoWriter(outName,cv2.VideoWriter_fourcc(*'DIVX'), 25, (width,height),0)
        
        numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Copying movie...')
        for idx,i in enumerate(np.arange(0,numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)))):
            print('Frame ' + str(idx) + ' of ' + str(len(np.arange(0,numFrames))))
            ret, im = vid.read()
            current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            vidOut.write(current)
        print('...Done copying!')
        vidOut.release()
        vid.release()
        return 1
    except: 
        try:
            vidOut.release()
            vid.release()
            return 0
        except:
            return -1

aviFile='S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Sleep_experiments/Larvae/220919_10_11_gria3xpo7/20220919-190935_Box1_0001.avi'
outName='D:/Data/SleepVid/20220919-190935_Box1_0001.avi'
outNameds='D:/Data/SleepVid/20220919-190935_Box1_SHORT.avi'
downsampleFrames=1000
downSampleVid(downsampleFrames, aviFile, outNameds)
copyVid(aviFile, outName)