# -*- coding: utf-8 -*-
"""
Track all fish in a social preference experiment

@author: dreostilab (Elena Dreosti)
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'D:\Tom\Github\SCZ_Fish\libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import shutil
import numpy as np
import matplotlib.pyplot as plt
import SCZ_utilities as SCZU
import SCZ_video as SCZV

# Specify Folder List and 
folderListFile = r'S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/FolderLists/Cohort/trioMiss.txt'
# ROI settings
individual_track_rois = True
individual_test_rois = True
individual_cue_rois = True

individual_rois=False
if individual_track_rois and individual_test_rois and individual_cue_rois: individual_rois=True
exp_date=''

# Set Flags
copy = False
preprocess = False
saveSummaryVid = False
analyze = True
endMins = 15 # only track the first 15 mins

# Read Folder List
groups, ages, folderNames, fishStatus, ROI_path = SCZU.read_folder_list1(folderListFile)

if copy:
    folderNames_orig=folderNames
    folderNames=[]
    for idx,folder in enumerate(folderNames_orig):
        src=folder
        _,_,_,name=folder.split(sep='\\',maxsplit=3)
        dstt=r'D:\\dataToTrack\\'
        dst=dstt+name
        print('Copying directory tree from ' + src + ' to new directory at ' + dstt)
        newDir=shutil.copytree(src, dst)
        folderNames.append(newDir)
        
# Bulk tracking of all folders in Folder List - preprocess first
if preprocess:
    for idx,folder in enumerate(folderNames):
        SCZV.process_video_summary_images_TR(folder,False, ROI_path=ROI_path, endMins = endMins, saveSummaryVid=saveSummaryVid)
        
if analyze:
    for idx,folder in enumerate(folderNames):
        status=fishStatus[idx]
        # Get Folder Names
        NS_folder, S_folder, _ = SCZU.get_folder_names(folder)
        # Load NS and S Test Crop Regions, as well as cue ROIs 
        # Decide whether we need individual ROI files for each video
        [track_ROIs, S_test_ROIs, S_stim_ROIs, cue_ROIs ]= SCZU.load_ROIs(folder,ROI_path=ROI_path, indFlag=individual_rois, report=True)
        report=False
        # Determine Fish Status       
        output_folder=NS_folder + '/Figures'
        SCZU.cycleMkDir(output_folder)
        print('Processing Non-Social fish ' + folder)
        fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SCZV.improved_fish_tracking(NS_folder, output_folder, track_ROIs, report=report, status=status, endMins=endMins)
        
        # Save Tracking (NS)
        for i in range(0,6):
            # Save NS
            path = NS_folder + r'/Tracking'
            SCZU.cycleMkDir(path)
            filename = path + r'/tracking' + str(i+1) + r'.npz'
            fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
            np.savez(filename, tracking=fish.T)
        
        # ---------------------
        # Process Video (S)
        output_folder=S_folder + '/Figures'
        SCZU.cycleMkDir(output_folder)
        print('Processing Social fish')
        # fxS, fyS, bxS, byS, exS, eyS, tailSegXS,tailSegYS,areaS, ortS, motS, failedAviFiles, errF = SCZV.arena_fish_tracking(aviFile, figureDirPath, track_ROIs, plot=1, cropOp=1, FPS=FPS, larvae=False)
        fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SCZV.improved_fish_tracking(S_folder, output_folder, track_ROIs, report=report,status=status, endMins=endMins)
    
        # Save Tracking (S)
        for i in range(0,6):
            # Save S_test
            path = S_folder + r'/Tracking'
            SCZU.cycleMkDir(path)
            filename = path + r'/tracking' + str(i+1) + r'.npz'
            fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
            np.savez(filename, tracking=fish.T)
        
        # Close Plots
        plt.close('all')
        print('FIN')
    #FIN
