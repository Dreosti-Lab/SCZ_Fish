# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:02:23 2023

@author: Tom
"""

lib_path = r'D:\Tom\Github\SCZ_Fish\libs'
import sys
sys.path.append(lib_path)

import cv2
import glob
import numpy as np
import SCZ_utilities as SCZU
import SCZ_video as SCZV
import BONSAI_ARK

def create_roi_grid_from_corners(top_left_roi, bottom_right_roi, num_rows=8, num_cols=12):
    # Importantly, ROIs are generated by row then column, i.e. the top left ROI is ROIs[0], the second ROI along the top from left to right is ROIs[1] and so on. 
    
    ROIs = []
    top_left_x, top_left_y, roi_width, roi_height = top_left_roi
    bottom_right_x, bottom_right_y, _, _ = bottom_right_roi

    x_spacing = (bottom_right_x - top_left_x) / (num_cols - 1)
    y_spacing = (bottom_right_y - top_left_y) / (num_rows - 1)

    for row in range(num_rows):
        for col in range(num_cols):
            x = top_left_x + col * x_spacing
            y = top_left_y + row * y_spacing
            ROIs.append((int(x), int(y), int(roi_width), int(roi_height)))
    return ROIs

def read_folder_list(folderListFile): 
## Read Folder List file 
    folderFile = open(folderListFile, "r") #"r" means read the file
    folderList = folderFile.readlines() # returns a list containing the lines
    data_path = folderList[0][:-1] # Read Data Path which is the first line
    folderList = folderList[1:] # Remove first line becasue it contains the path
    
    folderNames = [] # We use this becasue we do not know the exact length
    
    for i, f in enumerate(folderList):  #enumerate tells you what folder is 'i'
        stringLine = f[:].split()
        expFolderName = data_path + stringLine[0]
        folderNames.append(expFolderName)
        
    return data_path,folderNames


def load_TLBR_ROIs(ROI_path):
    bonsaiFiles = glob.glob(ROI_path + '/*.bonsai')
    bonsaiFile = bonsaiFiles[0]
    
    ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFile)
    
    # Find the ROI with the lowest value of the x_origin
    top_left_roi = ROIs[np.argmin(ROIs[:, 0])]
    
    # Find the other ROI as top_right_roi
    bottom_right_roi = [roi for roi in ROIs if not np.array_equal(roi, top_left_roi)][0]
    
    return top_left_roi, bottom_right_roi