# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:44:40 2013

@author: dreostilab (Elena Dreosti)
"""
# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import math
import glob
import cv2

lib_path = r'S:\WIBR_Dreosti_Lab\Tom\Github\Lonely_Fish_TR\Libraries'
TR_lib_path = r'S:\WIBR_Dreosti_Lab\Tom\Github\Arena_Zebrafish\libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)
sys.path.append(TR_lib_path)

import BONSAI_ARK
# import AZ_utilities as SCZU
import SCZ_utilities as SCZU

# Utilities for processing videos of Social Experiments
def drawROI(img,roi,color=[255,255,255],weight=10):
    
    tl = (int(roi[0]),int(roi[1]))
    tr = (int(roi[0]+roi[2]), int(roi[1]))
    bl = (int(roi[0]), int(roi[1]+roi[3]))
    br = (int(roi[0]+roi[2]),int(roi[1]+roi[3]))
    
     # draw the lines            
    cv2.line(img,tl,tr,color,thickness=weight)
    cv2.line(img,tl,bl,color,thickness=weight)
    cv2.line(img,tr,br,color,thickness=weight)
    cv2.line(img,bl,br,color,thickness=weight)
        
    return img
    
# Process Video : Make Summary Images TR folder structure
def process_video_summary_images_TR(folder, social, ROI_path='',saveSummaryVid=True,save=True, endMins = -1):
    
    output_folder=folder+'/ROI_Figures'
    SCZU.cycleMkDir(output_folder)
    _, S_folder, ROI_folder = SCZU.get_folder_names(folder)
    aviFiles = glob.glob(S_folder+'/*.avi')
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    
    if endMins < 0:
        numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        frameRate = int(vid.get(5))
        numFrames = frameRate * (endMins * 60)
        
    ## Show Crop Regions
    # Load Tracking Crop Regions
    if len(ROI_path)==0:
        ROI_folder=ROI_path
        
    bonsaiFiles = glob.glob(ROI_folder+'/*_track.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    ROI = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    track_ROIs=ROI[:, :]
    
    # Load Cue Crop Regions
    bonsaiFiles = glob.glob(ROI_folder+'/*_cue.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    ROI = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    cue_ROIs=ROI[:,:]
    
    # Load NS Crop Regions
    bonsaiFiles = glob.glob(ROI_folder+'/*_NS.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    ROI = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    NS_ROIs=ROI[:,:]
    
    # Load S Crop Regions
    bonsaiFiles = glob.glob(ROI_folder+'/*_S.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    ROI = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    S_ROIs=ROI[:, :]
    
    # Pull an image and draw the ROIs
    img=SCZU.grabFrame(aviFile,500)
    img=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(0,6):
    
        # draw ROIs
        img=drawROI(img,track_ROIs[i,:],color=(0,255,50),weight=10)
        img=drawROI(img,cue_ROIs[i,:],color=(255,50,255),weight=10)
        img=drawROI(img,S_ROIs[i,:],color=(0,50,255),weight=10)
        img=drawROI(img,NS_ROIs[i,:],color=(255,50,0),weight=10)
        
    if save:
        savename=output_folder+'/ROIs.png'
        cv2.imwrite(savename,img)


     # Read First Frame
    SCZU.setFrame(vid,100)
    ret, im = vid.read()
    previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    width = np.size(previous, 1)
    height = np.size(previous, 0)
    
    # Alloctae Image Space
    #stepFrames = 250 # Add a background frame every 2.5 seconds for 50 seconds
    #bFrames = 20
    stepFrames = 1500 # Add a background frame every 15 seconds for 600 seconds
    bFrames = 40
    accumulated_diff = np.zeros((height, width), dtype = float)
    backgroundStack = np.zeros((height, width, bFrames), dtype = float)
    background = np.zeros((height, width), dtype = float)
    croppedTest = np.zeros((height, width), dtype = float)
    croppedStim = np.zeros((height, width), dtype = float)
    bCount = 0
    for i, f in enumerate(range(0, numFrames, stepFrames)):
        
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        absDiff = cv2.absdiff(previous, current)
        level, threshold = cv2.threshold(absDiff,20,255,cv2.THRESH_TOZERO)
        previous = current
       
        # Accumulate differences
        accumulated_diff = accumulated_diff + threshold
                
        # Add to background stack
        if(bCount < bFrames):
            backgroundStack[:,:,bCount] = current
            bCount = bCount + 1
        
        # print (numFrames-f)
    norm_max=np.max(accumulated_diff)
    # Normalize accumulated difference image
    accumulated_diff = accumulated_diff/np.max(norm_max)
    accumulated_diff = np.ubyte(accumulated_diff*255)
    
    # Enhance Contrast (Histogram Equalize)
    equ = cv2.equalizeHist(accumulated_diff)
    
    # Compute Background Frame (median or mode)
    background = np.median(backgroundStack, axis = 2)
    
    summary = np.zeros((height, width, 3), dtype = float)
    if social:
        summary[:,:, 0] = croppedStim;
    
    summary[:,:, 1] = equ;
    summary[:,:, 2] = croppedTest;
    
    # misc.imsave(output_folder + r'/background_old.png', background)
    cv2.imwrite(output_folder + r'/summary.png', summary)    
#    cv2.imwrite(outputFolder + r'/background.png', cv2.fromarray(background))
    cv2.imwrite(output_folder + r'/background.png', background)
    # Using SciPy to save caused a weird rescaling when the images were dim.
    # This will change not only the background in the beginning but the threshold estimate
    
    
    if saveSummaryVid:
        SCZU.setFrame(vid,100)
        ret, im = vid.read()
        previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        stepFrames = 500 # Add a background frame every 5 seconds for 600 seconds
        numFrames=100*600
        savePath=output_folder+r'/summaryVideo.avi'
        print('Saving summary video at ' + str(savePath))
        vidOut= cv2.VideoWriter(savePath,cv2.VideoWriter_fourcc(*'DIVX'), 10, (height,width),0)
        accumulated_diff = np.zeros((height, width), dtype = float)
        for i, f in enumerate(range(0, numFrames, stepFrames)):
#            print(f)
            vid.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, im = vid.read()
            current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            absDiff = cv2.absdiff(previous, current)
            level, threshold = cv2.threshold(absDiff,20,255,cv2.THRESH_TOZERO)
            previous=current
            # Accumulate differences
            accumulated_diff = accumulated_diff + threshold
            accumulated_diff_frame = accumulated_diff/norm_max
            accumulated_diff_frame = np.ubyte(accumulated_diff_frame*255)
            equ = np.uint8(cv2.equalizeHist(accumulated_diff_frame))
            acc_summ_vid_frame = np.zeros((height, width, 3), dtype = 'uint8')
            
            acc_summ_vid_frame[:,:, 1] = equ;
#            cv2.imwrite(output_folder + r'/summary_frames/summary_frame_' + str(i)+'.png', acc_summ_vid_frame)    
            vidOut.write(acc_summ_vid_frame)
            
        print('Finished making movie')
        vidOut.release()
        
        
    vid.release()
    

    return 0
    
# Process Video : Make Summary Images
def pre_process_video_summary_images(folder, social):
    
    # Load Video
    aviFiles = glob.glob(folder+'/*.avi')
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read First Frame
    ret, im = vid.read()
    previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    width = np.size(previous, 1)
    height = np.size(previous, 0)
    
    # Alloctae Image Space
    stepFrames = 250 # Add a background frame every 2.5 seconds for 50 seconds
    bFrames = 50
    thresholdValue=10
    accumulated_diff = np.zeros((height, width), dtype = float)
    backgroundStack = np.zeros((height, width, bFrames), dtype = float)
    background = np.zeros((height, width), dtype = float)
    bCount = 0
    for i, f in enumerate(range(0, numFrames, stepFrames)):
        
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        absDiff = cv2.absdiff(previous, current)
        level, threshold = cv2.threshold(absDiff,thresholdValue,255,cv2.THRESH_TOZERO)
        previous = current
       
        # Accumulate differences
        accumulated_diff = accumulated_diff + threshold
        
        # Add to background stack
        if(bCount < bFrames):
            backgroundStack[:,:,bCount] = current
            bCount = bCount + 1
        
        print (numFrames-f)
#        print (bCount)

    vid.release()

    # Normalize accumulated difference image
    accumulated_diff = accumulated_diff/np.max(accumulated_diff)
    accumulated_diff = np.ubyte(accumulated_diff*255)
    
    # Enhance Contrast (Histogram Equalize)
    equ = cv2.equalizeHist(accumulated_diff)

    # Compute Background Frame (median or mode)
    background = np.median(backgroundStack, axis = 2)

    saveFolder = folder
    misc.imsave(saveFolder + r'/difference.png', equ)    
    cv2.imwrite(saveFolder + r'/background.png', background)
    # Using SciPy to save caused a weird rescaling when the images were dim.
    # This will change not only the background in the beginning but the threshold estimate

    return 0

# Process Video : Make Summary Images
def process_video_summary_images(folder, social):
    
    # Load Video
    aviFiles = glob.glob(folder+'/*.avi')
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read First Frame
    ret, im = vid.read()
    previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    width = np.size(previous, 1)
    height = np.size(previous, 0)
    
    # Alloctae Image Space
    #stepFrames = 250 # Add a background frame every 2.5 seconds for 50 seconds
    #bFrames = 20
    stepFrames = 1500 # Add a background frame every 15 seconds for 600 seconds
    bFrames = 40
    accumulated_diff = np.zeros((height, width), dtype = float)
    backgroundStack = np.zeros((height, width, bFrames), dtype = float)
    background = np.zeros((height, width), dtype = float)
    croppedTest = np.zeros((height, width), dtype = float)
    croppedStim = np.zeros((height, width), dtype = float)
    bCount = 0
    for i, f in enumerate(range(0, numFrames, stepFrames)):
        
        vid.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, im = vid.read()
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        absDiff = cv2.absdiff(previous, current)
        level, threshold = cv2.threshold(absDiff,20,255,cv2.THRESH_TOZERO)
        previous = current
       
        # Accumulate differences
        accumulated_diff = accumulated_diff + threshold
        
        # Add to background stack
        if(bCount < bFrames):
            backgroundStack[:,:,bCount] = current
            bCount = bCount + 1
        
        print (numFrames-f)

    vid.release()

    # Normalize accumulated difference image
    accumulated_diff = accumulated_diff/np.max(accumulated_diff)
    accumulated_diff = np.ubyte(accumulated_diff*255)
    
    # Enhance Contrast (Histogram Equalize)
    equ = cv2.equalizeHist(accumulated_diff)

    # Compute Background Frame (median or mode)
    background = np.median(backgroundStack, axis = 2)
    
    # Maybe Display Background
    
    plt.figure()    
    plt.imshow(background, cmap = plt.cm.gray, vmin = 0, vmax = 255)
    plt.draw()
    plt.pause(0.001)
    
    ## Show Crop Regions
    
    # Load Test Crop Regions
    bonsaiFiles = glob.glob(folder+'/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    test_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    test_ROIs = test_ROIs[:, :]
    croppedTest = np.copy(background)
    
    # Load Stim Crop Regions
    if social:
        bonsaiFiles = glob.glob(folder+'/Social_Fish/*.bonsai')
        bonsaiFiles = bonsaiFiles[0]
        stim_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
        stim_ROIs = stim_ROIs[:, :]
        croppedStim = np.copy(background)    
    
    for i in range(0,6):
        r1 = np.int(test_ROIs[i, 1])
        r2 = np.int(r1+test_ROIs[i, 3])
        c1 = np.int(test_ROIs[i, 0])
        c2 = np.int(c1+test_ROIs[i, 2])
        croppedTest[r1:r2, c1:c2] = 0
        
        
        if social:
            r1 = np.int(stim_ROIs[i, 1])
            r2 = np.int(r1+stim_ROIs[i, 3])
            c1 = np.int(stim_ROIs[i, 0])
            c2 = np.int(c1+stim_ROIs[i, 2])
            croppedStim[r1:r2, c1:c2] = 0
    
    summary = np.zeros((height, width, 3), dtype = float)
    if social:
        summary[:,:, 0] = croppedStim;
    
    summary[:,:, 1] = equ;
    summary[:,:, 2] = croppedTest;
    
    saveFolder = folder
    misc.imsave(saveFolder + r'/background_old.png', background)
    misc.imsave(saveFolder + r'/summary.png', summary)    
#    cv2.imwrite(saveFolder + r'/background.png', cv2.fromarray(background))
    cv2.imwrite(saveFolder + r'/background.png', background)
    # Using SciPy to save caused a weird rescaling when the images were dim.
    # This will change not only the background in the beginning but the threshold estimate

    return 0

# Compute the initial background for each ROI
def compute_intial_backgrounds(vid, ROIs):

    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)

    # Allocate space for all ROI backgrounds
    background_ROIs = []
    for i in range(0,6):
        w, h = get_ROI_size(ROIs, i)
        background_ROIs.append(np.zeros((h, w), dtype = np.float32))
    
    # Find initial background for each ROI
    for i in range(0,6):
        # Allocate space for background stack
        crop_width, crop_height = get_ROI_size(ROIs, i)
        stepFrames = 1000 # Check background frame every 10 seconds
        bFrames = 20
        backgroundStack = np.zeros((crop_height, crop_width, bFrames), dtype = np.float32)
        previous = np.zeros((crop_height, crop_width), dtype = np.float32)
        
        # Store first frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, im = vid.read()
        current = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        crop, xOff, yOff = get_ROI_crop(current, ROIs, i)
        backgroundStack[:,:,0] = np.copy(crop)
        previous = np.copy(crop)
        bCount = 1
        
        # Search for useful background frames (significantly different than previous)
        changes = []
        for f in range(stepFrames, numFrames, stepFrames):
            
            # Read frame
            vid.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, im = vid.read()
            current = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            crop, xOff, yOff = get_ROI_crop(current, ROIs, i)
        
            # Measure change from current to previous frame
            absdiff = np.abs(previous-crop)
            level = np.median(crop)/7
            change = np.mean(absdiff > level)
            changes.append(change)
            previous = np.copy(crop)
            
            # If significant, add to stack...possible finish
            if(change > 0.0075):
                backgroundStack[:,:,bCount] = np.copy(crop)
                bCount = bCount + 1
                if(bCount == bFrames):
                    print("Background for ROI(" + str(i) + ") found on frame " + str(f))
                    break
        
        # Compute background
        backgroundStack = backgroundStack[:,:, 0:bCount]
        background_ROIs[i] = np.median(backgroundStack, axis=2)
                        
    # Return initial background
    return background_ROIs
#------------------------------------------------------------------------------
# Process Video : Track fish in AVI
def improved_fish_tracking(input_folder, output_folder, ROIs, report=True, status=[1,1,1,1,1,1], endMins=-1):
    
    # Load Video
    aviFiles = glob.glob(input_folder+'/*.avi')
    aviFile = aviFiles[0]
    print('Loading ' + aviFile)
    import time
    tic = time.time()
    vid = cv2.VideoCapture(aviFile)
    toc = time.time() - tic
    print('Done loading... took' + str(toc) + ' seconds')
    # Compute a "Starting" Background
    # - Median value of 20 frames with significant difference between them
    print('Finding background')
    background_ROIs = compute_intial_backgrounds(vid, ROIs)
    
    # Algorithm
    # 1. Find initial background guess for each ROI
    # 2. Extract Crop regions from ROIs
    # 3. Threshold ROI using median/7 of each crop region, Binary Close image using 5 rad disc
    # 4. Find largest particle (Contour)
    # 5. - Compute Weighted Centroid (X,Y) for Eye Region (10% of brightest pixels)
    # 6. - Compute Binary Centroid of Body Region (50% of brightest pixels - eyeRegion)
    # 7. - Compute Heading
    
    
    if endMins < 0:
        numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)
    else:
        frameRate = int(vid.get(5))
        numFrames = frameRate * (endMins * 60)
        
    # numFrames = (15*60)*120
    # Allocate ROI (crop region) space
    previous_ROIs = []
    for i in range(0,6):
        w, h = get_ROI_size(ROIs, i)
        previous_ROIs.append(np.zeros((h, w), dtype = np.uint8))
    
    # Allocate Tracking Data Space
    fxS = np.zeros((numFrames,6))           # Fish X
    fyS = np.zeros((numFrames,6))           # Fish Y
    bxS = np.zeros((numFrames,6))           # Body X
    byS = np.zeros((numFrames,6))           # Body Y
    exS = np.zeros((numFrames,6))           # Eye X
    eyS = np.zeros((numFrames,6))           # Eye Y
    areaS = np.zeros((numFrames,6))         # area (-1 if error)
    ortS = np.zeros((numFrames,6))          # heading/orientation (angle from body to eyes)
    motS = np.zeros((numFrames,6))          # frame-by-frame change in segmented particle
        
    # Track within each ROI
    plt.figure(figsize=(8,6))
    
    # Skip first (sometimes corrupt) 100 frames (1 sec)
    SCZU.setFrame(vid,100)
    for f in range(0,numFrames): 
        # Read next frame        
        ret, im = vid.read()
        
        # Convert to grayscale (uint8)
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        
        # Process each ROI
        for i in range(0,6):
            # print('Processing ROI ' + str(i+1))            
            # Extract Crop Region
            crop, xOff, yOff = get_ROI_crop(current, ROIs, i)
            crop_height, crop_width = np.shape(crop)

            # Difference from current background
            diff = background_ROIs[i] - crop
            
            # Determine current threshold
            threshold_level = np.median(diff)+(3*np.std(diff))           
   
            # Threshold            
            level, threshold = cv2.threshold(diff,threshold_level,255,cv2.THRESH_BINARY)
            
            # Convert to uint8
            threshold = np.uint8(threshold)
            
            # Binary Close
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
            # Find Binary Contours            
            contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            
            # Create Binary Mask Image
            mask = np.zeros(crop.shape,np.uint8)
                       
            # If there are NO contours, then skip tracking
            if len(contours) == 0:
                if f!= 0:
                    area = -1.0
                    fX = fxS[f-1, i] - xOff
                    fY = fyS[f-1, i] - yOff
                    bX = bxS[f-1, i] - xOff
                    bY = byS[f-1, i] - yOff
                    eX = exS[f-1, i] - xOff
                    eY = eyS[f-1, i] - yOff
                    heading = ortS[f-1, i]
                    motion = -1.0
                else:
                    area = -1.0
                    fX = xOff
                    fY = yOff
                    bX = xOff
                    bY = yOff
                    eX = xOff
                    eY = yOff
                    heading = -181.0
                    motion = -1.0
            
            else:
                # Get Largest Contour (fish, ideally)
                largest_cnt, area = get_largest_contour(contours)
                
                # If the particle to too small to consider, skip frame
                if area == 0.0:
                    if f!= 0:
                        fX = fxS[f-1, i] - xOff
                        fY = fyS[f-1, i] - yOff
                        bX = bxS[f-1, i] - xOff
                        bY = byS[f-1, i] - yOff
                        eX = exS[f-1, i] - xOff
                        eY = eyS[f-1, i] - yOff
                        heading = ortS[f-1, i]
                        motion = -1.0
                    else:
                        area = -1.0
                        fX = xOff
                        fY = yOff
                        bX = xOff
                        bY = yOff
                        eX = xOff
                        eY = yOff
                        heading = -181.0
                        motion = -1.0
                        
                else:
                    # Draw contours into Mask Image (1 for Fish, 0 for Background)
                    cv2.drawContours(mask,[largest_cnt],0,1,-1) # -1 draw the contour filled
                    pixelpoints = np.transpose(np.nonzero(mask))
                    
                    # Get Area (again)
                    area = np.size(pixelpoints, 0)
                    
                    # ---------------------------------------------------------------------------------
                    # Compute Frame-by-Frame Motion (absolute changes above threshold)
                    # - Normalize by total absdiff from background
                    if (f != 0):
                        absdiff = np.abs(diff)
                        absdiff[absdiff < threshold_level] = 0
                        totalAbsDiff = np.sum(np.abs(absdiff))
                        frame_by_frame_absdiff = np.abs(np.float32(previous_ROIs[i]) - np.float32(crop)) / 2 # Adjust for increases and decreases across frames
                        frame_by_frame_absdiff[frame_by_frame_absdiff < threshold_level] = 0
                        motion = np.sum(np.abs(frame_by_frame_absdiff))/totalAbsDiff
                    else:
                        motion = 0
                    
                    # Save Masked Fish Image from ROI (for subsequent frames motion calculation)
                    previous_ROIs[i] = np.copy(crop)
                    
                    # ---------------------------------------------------------------------------------
                    # Find Body and Eye Centroids
                    area = np.float(area)
                    
                    # Highlight 50% of the birghtest pixels (body + eyes)                    
                    numBodyPixels = np.int(np.ceil(area/2))
                    
                    # Highlight 10% of the birghtest pixels (mostly eyes)     
                    numEyePixels = np.int(np.ceil(area/10))
                    
                    # Fish Pixel Values (difference from background)
                    fishValues = diff[pixelpoints[:,0], pixelpoints[:,1]]
                    sortedFishValues = np.sort(fishValues)
                    
                    bodyThreshold = sortedFishValues[-numBodyPixels]                    
                    eyeThreshold = sortedFishValues[-numEyePixels]

                    # Compute Binary/Weighted Centroids
                    r = pixelpoints[:,0]
                    c = pixelpoints[:,1]
                    all_values = diff[r,c]
                    all_values = all_values.astype(float)
                    r = r.astype(float)
                    c = c.astype(float)
                    
                    # Fish Centroid
                    values = np.copy(all_values)
                    values = (values-threshold_level+1)
                    acc = np.sum(values)
                    fX = np.float(np.sum(c*values))/acc
                    fY = np.float(np.sum(r*values))/acc
                    
                    # Eye Centroid (a weighted centorid)
                    values = np.copy(all_values)                   
                    values = (values-eyeThreshold+1)
                    values[values < 0] = 0
                    acc = np.sum(values)
                    eX = np.float(np.sum(c*values))/acc
                    eY = np.float(np.sum(r*values))/acc
    
                    # Body Centroid (a binary centroid, excluding "eye" pixels)
                    values = np.copy(all_values)                   
                    values[values < bodyThreshold] = 0
                    values[values >= bodyThreshold] = 1                                                            
                    values[values > eyeThreshold] = 0                                                            
                    acc = np.sum(values)
                    bX = np.float(np.sum(c*values))/acc
                    bY = np.float(np.sum(r*values))/acc
                    
                    # ---------------------------------------------------------------------------------
                    # Heading (0 deg to right, 90 deg up)
                    if (bY != eY) or (eX != bX):
                        heading = math.atan2((bY-eY), (eX-bX)) * (360.0/(2*np.pi))
                    else:
                        heading = -181.00
            
            # ---------------------------------------------------------------------------------
            # Store data in arrays
            
            # Shift X,Y Values by ROI offset and store in Matrix
            fxS[f, i] = fX + xOff
            fyS[f, i] = fY + yOff
            bxS[f, i] = bX + xOff
            byS[f, i] = bY + yOff
            exS[f, i] = eX + xOff
            eyS[f, i] = eY + yOff
            areaS[f, i] = area
            ortS[f, i] = heading
            motS[f, i] = motion
            
            # -----------------------------------------------------------------
            # Update this ROIs background estimate (everywhere except the (dilated) Fish)
            current_background = np.copy(background_ROIs[i])            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
            dilated_fish = cv2.dilate(mask, kernel, iterations = 2)           
            updated_background = (np.float32(crop) * 0.01) + (current_background * 0.99)
            updated_background[dilated_fish==1] = current_background[dilated_fish==1]            
            background_ROIs[i] = np.copy(updated_background)
            
            
        # ---------------------------------------------------------------------------------
        # Plot All Fish in Movie with Tracking Overlay
        if (f == 0) or (f == numFrames-1):
            plt.clf()
            enhanced = cv2.multiply(current, 1)
            color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            plt.imshow(color)
            plt.axis('image')
            for i in range(0,6):
                plt.plot(fxS[f, i],fyS[f, i],'b.', markersize = 1)
                plt.plot(exS[f, i],eyS[f, i],'r.', markersize = 3)
                plt.plot(bxS[f, i],byS[f, i],'co', markersize = 3)
                plt.text(bxS[f, i]+10,byS[f, i]+10,  '{0:.1f}'.format(ortS[f, i]), color = [1.0, 1.0, 0.0, 0.5])
                plt.text(bxS[f, i]+10,byS[f, i]+30,  '{0:.0f}'.format(areaS[f, i]), color = [1.0, 0.5, 0.0, 0.5])
            plt.draw()
            plt.pause(0.001)
            
        # ---------------------------------------------------------------------------------
        # Save Tracking Summary
        if(f == 0):
            plt.savefig(output_folder+'/initial_tracking.png', dpi=300)
            plt.figure('backgrounds')
            for i in range(0,6):
                plt.subplot(2,3,i+1)
                plt.imshow(background_ROIs[i])
            plt.savefig(output_folder+'/initial_backgrounds.png', dpi=300)
            plt.close('backgrounds')
        if(f == numFrames-1):
            plt.savefig(output_folder+'/final_tracking.png', dpi=300)
            plt.figure('backgrounds')
            for i in range(0,6):
                plt.subplot(2,3,i+1)
                plt.imshow(background_ROIs[i])
            plt.savefig(output_folder+'/final_backgrounds.png', dpi=300)
            plt.close('backgrounds')

        # Report Progress
        if report:
            if (f%120) == 0:
                bs = '\b' * 1000            # The backspace
                print(bs)
                print (numFrames-f)
    
    # -------------------------------------------------------------------------
    # Close Video File
    vid.release()
    
    # Return tracking data
    return fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS
#------------------------------------------------------------------------------
 
# Return cropped image from ROI list
def get_ROI_crop(image, ROIs, numROi):
    r1 = np.int(ROIs[numROi, 1])
    r2 = np.int(r1+ROIs[numROi, 3])
    c1 = np.int(ROIs[numROi, 0])
    c2 = np.int(c1+ROIs[numROi, 2])
    crop = image[r1:r2, c1:c2]
    
    return crop, c1, r1
    
# Return ROI size from ROI list
def get_ROI_size(ROIs, numROi):
    width = int(ROIs[numROi, 2])
    height = int(ROIs[numROi, 3])
    
    return width, height

# Return largest (area) cotour from contour list
def get_largest_contour(contours):
    # Find contour with maximum area and store it as best_cnt
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
    if max_area > 0:
        return best_cnt, max_area
    else:
        return cnt, max_area

# Compare Old (scaled) and New (non-scaled) background images
def compare_backgrounds(folder):

    # Load -Initial- Background Frame (histogram from first 50 seconds)
    backgroundFile = folder + r'/background_old.png'
    background_old = misc.imread(backgroundFile, False)
    
    backgroundFile = folder + r'/background.png'
    background = misc.imread(backgroundFile, False)
    absDiff = cv2.absdiff(background_old, background)

    return np.mean(absDiff)

# FIN
