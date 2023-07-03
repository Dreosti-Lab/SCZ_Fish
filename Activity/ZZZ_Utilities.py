# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:22:34 2022

@author: Tom
"""
lib_path = r'D:\Tom\Github\SCZ_Fish\libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)

import SCZ_video as SCZV
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from PIL import Image
# from skimage.feature import canny
# from skimage.transform import hough_line, hough_line_peaks
# import math
import time




    
def calculate_grid_rectangles(x, y, w, h, c=12, r=8):
    
    # Calculate the spacing between ROIs
    s_x = w
    s_y = h

    rects = []

    for row in range(r):
        for col in range(c):
            # Calculate the top left corner of each rectangle in the grid
            rect_x = x + col * s_x
            rect_y = y + row * s_y

            # Calculate the bottom right corner of each rectangle (width and height)
            rect_w = s_x
            rect_h = s_y
            rects.append((rect_x, rect_y, rect_w, rect_h))

    return np.array(rects)

def warp_grid_rectangles(rects):
    # Get the dimensions of the first rectangle (corner rectangle)
    corner_x, corner_y, corner_w, corner_h = rects[0]

    warped_rects = []

    for rect in rects:
        rect_x, rect_y, rect_w, rect_h = rect

        # Calculate the variation in width and height based on distance from the corner rectangle
        diff_w = abs(rect_x - corner_x) / corner_w * 6
        diff_h = abs(rect_y - corner_y) / corner_h * 6

        # Adjust the width and height of each rectangle based on the variations
        warped_w = rect_w + diff_w
        warped_h = rect_h + diff_h

        warped_rects.append((rect_x, rect_y, warped_w, warped_h))

    return np.array(warped_rects)

def create_rectangle_masks(image, rectangles):
    masks = []
    for rect in rectangles:
        x, y, w, h = rect
        mask = np.zeros_like(image, dtype=bool)
        mask[int(y):int(y)+int(h), int(x):int(x)+int(w)] = True
        masks.append(mask)
    return masks

def convert_mask_to_points(mask):
    points = []
    
    # Find the coordinates of the non-zero elements in the mask
    coords = np.transpose(np.nonzero(mask))
    mask_points = [(coord[1], coord[0]) for coord in coords]
    points.append(mask_points)
    return np.array(points)

def rotate_masks_corner(masks, corner, angle):
    
    if corner == 'TL':
        origin = (masks[0].nonzero()[0].min(), masks[0].nonzero()[1].min())
    elif corner == 'TR':
        origin = (masks[11].nonzero()[0].min(), masks[0].nonzero()[1].max())
    elif corner == 'BL':
        origin = (masks[84].nonzero()[0].max(), masks[0].nonzero()[1].min())
    elif corner == 'BR':
        origin = (masks[95].nonzero()[0].max(), masks[0].nonzero()[1].max())
    else:
        raise ValueError("Invalid corner position. Valid positions are 'TL', 'TR', 'BL', and 'BR'.")
    origin=(int(origin[0]),int(origin[1]))
    rotated_masks = []
    for mask in masks:
        # Convert the boolean mask to a binary image (0s and 255s)
        points=convert_mask_to_points(mask)

        # Find the rotated rectangle around the origin
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Rotate the rectangle by the specified angle around the origin
        rotation_matrix = cv2.getRotationMatrix2D(origin, angle, 1.0)
        rotated_box = cv2.transform(np.array([box]), rotation_matrix)[0]

        # Create a rotated mask from the rotated rectangle
        rotated_mask = np.zeros(mask.shape)
        cv2.fillPoly(rotated_mask, [rotated_box], 1)

        # Append the rotated mask to the list
        rotated_masks.append(rotated_mask)

    return rotated_masks

def draw_masks(masks, im):
    
    for mask in masks:
        # Find the contours of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours (outlines) on the image
        cv2.drawContours(im, contours, -1, (0, 255, 0), thickness=2)

    # Display the image with mask outlines
    plt.imshow(im)
    plt.axis('off')
    plt.show()
    
def draw_rectangles(rects,im):

    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(im)
    # draw the rectangles as patches
    for rect in rects:
        ROI = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(ROI)

    ax.set_xlim(0, im.shape[1])
    ax.set_ylim(0, im.shape[0])


#%%
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
        
def trimVid(startFrame,lengthFrame, vid, outName):
    try:
        vid.set(cv2.CAP_PROP_POS_FRAMES, 99) # skip first 5 seconds (sometimes some bright artifact)
        ret,im=vid.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        width = np.size(im, 1)
        height = np.size(im, 0)
        vidOut= cv2.VideoWriter(outName,cv2.VideoWriter_fourcc(*'DIVX'), 10, (width,height),0)
        # stepFrame=int(np.floor(np.divide(numFrames_orig,downsampleFrames)))
        print('Trimming movie...')
        for idx,i in enumerate(np.arange(startFrame,startFrame+lengthFrame)):
            # print('Frame ' + str(idx) + ' of ' + str(len(np.arange(100,numFrames_orig, stepFrame))))
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, im = vid.read()
            current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            vidOut.write(current)
        print('...Done trimming!')
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
#%%
aviFile='S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/Sleep_experiments/Larvae/220926_10_11_SP4_G03_Con_Scram_G_Con/Box1/20220926-205905_Box1_0001.avi'
# aviFile='S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/Sleep_experiments/Larvae/220919_10_11_gria3xpo7/Box1/20220919-190935_Box1_0001.mp4'#'S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/Sleep_experiments/Larvae/220815_14_15_Gria3Trio/Box1/220815_14_15_Gria3Trio_Box1_0001.avi'
outName='D:/Data/SleepVid/220926_10_11_SP4_G03_Con_Scram_G_Con_Box1_SHORT.avi'#220815_14_15_Gria3Trio_Box1_0001_SHORT.avi'
print('Loading movie...')
tic=time.time()
vid=cv2.VideoCapture(aviFile)
toc=time.time()-tic
print('...Done loading! Took ' + str(toc) + ' seconds')
# outNameds='D:/Data/SleepVid/20220919-190935_Box1_SHORT.avi'
# downsampleFrames=1000
# downSampleVid(downsampleFrames, aviFile, outNameds)
# trimVid(100,25*60*20,vid,outName)
# copyVid(aviFile, outName)


#%%
vid=cv2.VideoCapture(outName)
ret,im=vid.read()
im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
x,y,w,h=34,43,88,88
rects = calculate_grid_rectangles(x, y, w, h)
# rects=warp_grid_rectangles(rects)
# masks=create_rectangle_masks(im,rects)
# corner = 'BL'
# angle=45
# masks=rotate_masks_corner(masks, corner, angle) - not that functional
# zrects=rotate_grid_rectangles(rects, corner, angle)
# draw_rectangles(zrects,im)
# draw_masks(masks,im)
folder='S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/Sleep_experiments/Larvae/220815_14_15_Gria3Trio/Box1/'
output_folder=folder+'Tracking'
# fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SCZV.improved_fish_tracking(folder, output_folder, rects)