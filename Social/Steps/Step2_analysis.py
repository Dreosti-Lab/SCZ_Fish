# -*- coding: utf-8 -*-
"""
Analyze all tracked fish in a social preference experiment

@author: Tom Ryan, UCL (Dreosti-Group) 
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'S:\WIBR_Dreosti_Lab\Tom\Github\SCZ_Model_Fish\libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)
#-----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
# Import local modules
import SCZ_utilities as SCZU
import SCZ_analysis as SCZA
import SCZ_summary as SCZS
import SCZ_bouts as SCZB

# %% Settings
# ROI settings
individual_track_rois = True
individual_test_rois = True
individual_cue_rois = True
if individual_track_rois and individual_test_rois and individual_cue_rois: individual_rois= True

# Specify Folder List
folderListFile = r'S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/FolderLists/testy.txt'
# folderListFile = r'S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/FolderLists/cumulative_' + gene + '_cohort_missing.txt' 
#folderListFile = 'S:/WIBR_Dreosti_Lab/Tom/Data/Lesion_Social/ShamCChamber.txt'

#base_path=r'S:/WIBR_Dreosti_Lab/Tom/Data/Lesion_Social/C-Chamber/Analysis'
base_path=r'S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/AnalysisRounds'
analysisRoot = base_path + r'/Analysis_TestNew/' 

#analysisFolder = base_path + r'/Sham'
# Set Flags
check = False
plot = True
resample = True # set to true to force resampling of movies that are not == FPS 
FPS = 100 
FPSs=[]

# Set freeze thresholds
freeze_threshold = 5*FPS # 5 seconds
long_freeze_threshold = 1*60*FPS # 1 minute
            
# Set motion thresholds
motionStartThreshold = 0.03
motionStopThreshold = 0.01

#%% Helper Functions

def parseTracking(tracking):
    
    fx = tracking[:,0] 
    fy = tracking[:,1]
    bx = tracking[:,2]
    by = tracking[:,3]
    ex = tracking[:,4]
    ey = tracking[:,5]
    area = tracking[:,6]
    ort = tracking[:,7]
    motion = tracking[:,8]
    
    return fx,fy,bx,by,ex,ey,area,ort,motion

def rescaleTracking(tracking, currentFPS, FPS, save=True, folder = None): 
    # resample all traces so they match FPS (not really needed if you are just using FPS for thresholds)
    [fx,fy,bx,by,ex,ey,area,ort,motion]=parseTracking(tracking)
    if currentFPS != FPS:
        fx=SCZU.remapToFreq(fx,currentFPS,FPS)
        fy=SCZU.remapToFreq(fy,currentFPS,FPS)
        bx=SCZU.remapToFreq(bx,currentFPS,FPS)
        by=SCZU.remapToFreq(by,currentFPS,FPS)
        ex=SCZU.remapToFreq(ex,currentFPS,FPS)
        ey=SCZU.remapToFreq(ey,currentFPS,FPS)
        area=SCZU.remapToFreq(area,currentFPS,FPS)
        ort=SCZU.remapToFreq(ort,currentFPS,FPS)
        motion=SCZU.remapToFreq(motion,currentFPS,FPS)
        
    tracking = np.vstack((fx, fy, bx, by, ex, ey, area, ort, motion)).T
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    if save:
        filename = folder + r'/Tracking/tracking' + str(i) + r'.npz'
        np.savez(filename, tracking=tracking)
        
    return tracking,currentFPS,FPS

#%% SCRIPT
# Read Folder List
groups, num, folderNames, fishStatus, ROI_path = SCZU.read_folder_list1(folderListFile)

# Bulk analysis of all folders
geneList=[]
idGene=0
for idx,folder in enumerate(folderNames):
    
    gene=folder.rsplit(sep='\\',maxsplit=3)[1]
    if gene not in geneList:
        geneList.append(gene)
    
    analysisFolder = analysisRoot + gene
    SCZU.cycleMkDir_forw(analysisFolder)
    
    # Get Folder Names
    NS_folder, S_folder, _ = SCZU.get_folder_names(folder)
                  
    # Load NS and S Test Crop Regions, as well as cue ROIs 
    # Decide whether we need individual ROI files for each video
    [track_ROIs, S_test_ROIs, S_stim_ROIs, cue_ROIs] = SCZU.load_ROIs(folder,ROI_path, indFlag=individual_rois, report=True)
    
    # Determine Fish Status       
    fishStat = fishStatus[idx, :]
    
    # Determine FPS for this fish set
    aviFiles = glob.glob(NS_folder+'/*.avi')
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    currentFPS=vid.get(cv2.CAP_PROP_FPS)
    vid.release()

    # ----------------------
    # Analyze and (maybe) plot each Fish
    for i in range(1,7):
        plotFilename = analysisFolder + '/' + str(int(groups[idx])) + '_' + str(int(num[idx])) + '_SPI_' + str(i) + '.png'  
        BoutFilename_ns = analysisFolder + '/' + str(int(groups[idx])) + '_' + str(int(num[idx])) + '_anglevsDist_NS' + str(i) + '.png'  
        BoutFilename_s = analysisFolder + '/' + str(int(groups[idx])) + '_' + str(int(num[idx])) + '_anglevsDist_S' + str(i) + '.png'  
        dataFilename = analysisFolder + '/' + str(int(groups[idx])) + '_' + str(int(num[idx])) + '_SUMMARY_' + str(i) + '.npz'
        # Only use "good" fish
        if fishStat[i-1] == 1:
            if check:
                # Check analysis doesn't already exist
                if os.path.exists(plotFilename) and os.path.exists(dataFilename):
                    print("Analysis data and figures already present for fish..." + gene + ' ' + str(int(groups[idx])) + '_' + str(int(num[idx])) + '_SUMMARY_' + str(i) + '.png ... skipping')
                    continue
            # Plot Fish Data (maybe)
            if plot:
                # Make a Figure per each fish
                plt.figure(figsize=(10, 12), dpi=150)
        
#           Get X and Y coordinates of ROI Test and Stim ROIs extremes 
            x_min = min(track_ROIs[i-1,0], S_stim_ROIs[i-1,0])
            y_min = min(track_ROIs[i-1,1], S_stim_ROIs[i-1,1])
            x_max = max(track_ROIs[i-1,0] + track_ROIs[i-1,2], cue_ROIs[i-1,0] + cue_ROIs[i-1,2])
            y_max = max(track_ROIs[i-1,1] + track_ROIs[i-1,3], cue_ROIs[i-1,1] + cue_ROIs[i-1,3])
            
            # ----------------------
            # Analyze NS - load tracking, resample if needed, and analyse
            ff = NS_folder
            
            trackingFile = ff + r'/Tracking/tracking' + str(i) + '.npz'
            data = np.load(trackingFile)
            tracking = data['tracking']
            fx,fy,bx,by,ex,ey,area,ort_raw,motion = parseTracking(tracking)      
            # Adjust orientation (0 is always facing the "stimulus" fish) - Depends on chamber
            ort = SCZU.adjust_ort_test(ort_raw, i)
            
            if resample:
                if currentFPS!=FPS:
                    filename = ff + r'/Tracking/tracking_raw' + str(i+1) + r'.npz' # save a raw version
                    np.savez(filename, tracking=tracking)
                    tracking_adj = np.vstack((fx, fy, bx, by, ex, ey, area, ort, motion)).T
                    tracking_adj,currentFPS,FPS = rescaleTracking(tracking_adj,currentFPS,FPS,folder=ff)     
                    fx,fy,bx,by,ex,ey,area,ort,motion = parseTracking(tracking_adj)        
                    tracking = tracking_adj
                    filename = ff + r'/Tracking/tracking_adj' + str(i+1) + r'.npz' # save a raw version
                    # np.savez(filename, tracking=tracking)
            
            # Compute VPI (NS)
            # By default uses half the tracking ROI to define Non-social and Social regions. 
            # Cue_ROIs is only used to determine the location of the stim (top/bottom, left/right)
            VPI_ns, AllVisibleFrames, AllNonVisibleFrames, VPI_ns_bins = SCZA.computeVPI(bx, by, track_ROIs[i-1], cue_ROIs[i-1], FPS) 

            # Compute SPI (NS)
            # By default uses one third of the height and one half of the width of tracking ROI to define Non-social and Social regions. 
            # Cue_ROIs is only used to determine the location of the stim (top/bottom, left/right)
            SPI_ns, AllSocialFrames_TF, AllNONSocialFrames_TF = SCZA.computeSPI(bx, by, track_ROIs[i-1], cue_ROIs[i-1])
            
            # Compute BPS (NS)
            BPS_ns, avgBout_ns = SCZS.measure_BPS(motion, motionStartThreshold, motionStopThreshold)
            
            # Compute Distance Traveled (NS)
            Distance_ns = SCZA.distance_traveled(bx, by, track_ROIs[i-1])
        
            # Compute Orientation Histograms (NS)
            OrtHist_ns_NonSocialSide = SCZS.ort_histogram(ort[AllNonVisibleFrames])
            OrtHist_ns_SocialSide = SCZS.ort_histogram(ort[AllVisibleFrames])
            
            # compute the number of mid crossings (NS)
            midCrossings_ns, _ =SCZA.computeMidCrossings(bx, by, track_ROIs[i-1], cue_ROIs[i-1], longhand=False)
            
            # Analyze "Bouts" amd "Pauses" (NS)
            Bouts_ns, Pauses_ns = SCZS.analyze_bouts_and_pauses(tracking, track_ROIs[i-1], cue_ROIs[i-1], AllVisibleFrames, motionStartThreshold, motionStopThreshold)
            
            # Analyze per bout metrics 
            boutsMet_ns=SCZB.analyze(tracking,path=False)
            # OUTPUT: 
            # bouts[i, 0] = starts[i] - 2 # 2 frames before Upper threshold crossing 
            # bouts[i, 1] = peaks[i]      # Peak frame
            # bouts[i, 2] = stops[i]+1    # frame of Lower threshold crossing
            # bouts[i, 3] = stops[i]-starts[i] # Durations
            # bouts[i, 4] = np.sum(speed_angle[starts[i]:stops[i]]) # Total angle change  
            # bouts[i, 5] = np.sqrt(sx[-1]*sx[-1] + sy[-1]*sy[-1]) # Net distance change
            # bouts[i, 6] = ax[-1]
            # bouts[i, 7] = ay[-1]
            boutDists_ns=boutsMet_ns[:,5]
            boutAngles_ns=boutsMet_ns[:,4]
            good_fish=True
            
            if len(Bouts_ns) < 2 : good_fish = False
            if good_fish:
                Percent_Moving_ns = 100 * np.sum(Bouts_ns[:,8])/len(motion)
                Percent_Paused_ns = 100 * np.sum(Pauses_ns[:,8])/len(motion)
            
                freezes_X,freezes_Y = [],[]
                # Count Freezes
                freezeBool=Pauses_ns[:,8] > freeze_threshold
                longFreezeBool=Pauses_ns[:,8] > long_freeze_threshold
                Freezes_ns = np.array(np.sum(freezeBool))
                Long_Freezes_ns = np.array(np.sum(longFreezeBool))
                Freezes_X_ns = Pauses_ns[:,1]
                Freezes_Y_ns = Pauses_ns[:,2]
            else:
                Percent_Moving_ns = 0
                Percent_Paused_ns = 0
            
                # Count Freezes
                Freezes_ns = 0
                Long_Freezes_ns = 0
                
                Freezes_X_ns = 0
                Freezes_Y_ns = 0

            # Plot NS (maybe)
            if plot and good_fish:
                # plot bout distance vs angle for this fish NS
                plt.figure('scatter')
                plt.scatter(boutAngles_ns,boutDists_ns,s=2,color='black',alpha=0.8)
                plt.xlim(-180,180)
                plt.ylim(-10,200)
                plt.ylabel('Net distance (mm)')
                plt.xlabel('Net angle change (deg)')
                plt.savefig(BoutFilename_ns,dpi=600)
                plt.close('scatter')
                
                plt.subplot(5,2,1)
                plt.axis('off')
                plt.plot(bx, by, '.', markersize=1, color = [0.0, 0.0, 0.0, 0.05])
                plt.plot(bx[AllSocialFrames_TF], by[AllSocialFrames_TF], '.', markersize=1, color = [1.0, 0.0, 0.0, 0.05], )
                plt.plot(bx[AllNONSocialFrames_TF], by[AllNONSocialFrames_TF], '.', markersize=1, color = [0.0, 0.0, 1.0, 0.05])
                plt.title('SPI: ' + format(SPI_ns, '.3f') + ', VPI: ' + format(VPI_ns, '.3f'))
                plt.axis([x_min, x_max, y_min, y_max])
                plt.gca().invert_yaxis()

                plt.subplot(5,2,3)
                plt.title('BPS: ' + format(BPS_ns, '.3f') + ', %Paused: ' + format(Percent_Paused_ns, '.2f') + ', %Moving: ' + format(Percent_Moving_ns, '.2f'))
                motion[motion == -1.0] = -0.01
                plt.axhline(motionStartThreshold, c="green")
                plt.axhline(motionStopThreshold, c="red")
                plt.plot(motion)

                plt.subplot(5,2,5)
                motion[motion == -1.0] = -0.01
                plt.axhline(motionStartThreshold, c="green")
                plt.axhline(motionStopThreshold, c="red")
                plt.plot(motion[50000:51000])

                plt.subplot(5,2,7)
                plt.plot(avgBout_ns, 'k')
                
                plt.subplot(5,2,9)
                plt.plot(area, 'r')

            # ----------------------
            # Analyze S
            ff = S_folder
            
            trackingFile = ff + r'/Tracking/tracking' + str(i) + '.npz'
            data = np.load(trackingFile)
            tracking = data['tracking']
            fx,fy,bx,by,ex,ey,area,ort,motion = parseTracking(tracking)  
            
            # Adjust orientation (0 is always facing the "stimulus" fish) - Depends on chamber
            ort = SCZU.adjust_ort_test(ort, i)

            if resample:
                if currentFPS!=FPS:
                    filename = ff + r'/Tracking/tracking_raw' + str(i+1) + r'.npz' # save a raw version
                    np.savez(filename, tracking=tracking)
                    tracking_adj = np.vstack((fx, fy, bx, by, ex, ey, area, ort, motion)).T
                    tracking_adj,currentFPS,FPS = rescaleTracking(tracking_adj,currentFPS,FPS,folder=ff)     
                    fx,fy,bx,by,ex,ey,area,ort,motion = parseTracking(tracking_adj)        
                    tracking = tracking_adj
                    
            # Compute VPI (S)
            VPI_s, AllVisibleFrames, AllNonVisibleFrames, VPI_s_bins = SCZA.computeVPI(bx, by, track_ROIs[i-1], cue_ROIs[i-1], FPS)

            # Compute SPI (S)
            SPI_s, AllSocialFrames_TF, AllNONSocialFrames_TF = SCZA.computeSPI(bx, by, track_ROIs[i-1], cue_ROIs[i-1])
            
            # Compute BPS (S)
            BPS_s, avgBout_s = SCZS.measure_BPS(motion, motionStartThreshold, motionStopThreshold)

            # Compute Distance Traveled (S)
            Distance_s = SCZA.distance_traveled(bx, by, track_ROIs[i-1])
            
            # Compute Orientation Histograms (S)
            OrtHist_s_NonSocialSide = SCZS.ort_histogram(ort[AllNonVisibleFrames])
            OrtHist_s_SocialSide = SCZS.ort_histogram(ort[AllVisibleFrames])
            
            # compute the number of mid crossings (NS)
            midCrossings_s, _ =SCZA.computeMidCrossings(bx, by, track_ROIs[i-1], cue_ROIs[i-1], longhand=False)
            
            # Analyze "Bouts" amd "Pauses" (NS)
            Bouts_s, Pauses_s = SCZS.analyze_bouts_and_pauses(tracking, track_ROIs[i-1], cue_ROIs[i-1], AllVisibleFrames, motionStartThreshold, motionStopThreshold)
           
            # Analyze per bout metrics 
            boutsMet_s=SCZB.analyze(tracking,path=False)
            # OUTPUT: 
            # bouts[i, 0] = starts[i] - 2 # 2 frames before Upper threshold crossing 
            # bouts[i, 1] = peaks[i]      # Peak frame
            # bouts[i, 2] = stops[i]+1    # frame of Lower threshold crossing
            # bouts[i, 3] = stops[i]-starts[i] # Durations
            # bouts[i, 4] = np.sum(speed_angle[starts[i]:stops[i]]) # Total angle change  
            # bouts[i, 5] = np.sqrt(sx[-1]*sx[-1] + sy[-1]*sy[-1]) # Net distance change
            # bouts[i, 6] = ax[-1]
            # bouts[i, 7] = ay[-1]
            boutDists_s=boutsMet_s[:,5]
            boutAngles_s=boutsMet_s[:,4]
            good_fish=True
            
            if len(Bouts_s) < 2 : good_fish = False
            if good_fish:
                Percent_Moving_s = 100 * np.sum(Bouts_s[:,8])/len(motion)
                Percent_Paused_s = 100 * np.sum(Pauses_s[:,8])/len(motion)
            
                # Count Freezes
                freezeBool=Pauses_s[:,8] > freeze_threshold
                longFreezeBool=Pauses_s[:,8] > long_freeze_threshold
                Freezes_s = np.array(np.sum(freezeBool))
                Long_Freezes_s = np.array(np.sum(longFreezeBool))
                Freezes_X_s = Pauses_s[:,1]
                Freezes_Y_s = Pauses_s[:,2]
            else:
                Percent_Moving_s = 0
                Percent_Paused_s = 0
            
                # Count Freezes
                Freezes_s = 0
                Long_Freezes_s = 0
                
                Freezes_X_s = 0
                Freezes_Y_s = 0
             
            # PLot S (maybe)
            if plot and good_fish:
                # plot bout distance vs angle for this fish S
                plt.figure('scatter')
                plt.scatter(boutAngles_s,boutDists_s,s=2,color='black',alpha=0.8)
                plt.xlim(-180,180)
                plt.ylim(-10,200)
                plt.ylabel('Net distance (mm)')
                plt.xlabel('Net angle change (deg)')
                plt.savefig(BoutFilename_s,dpi=600)
                plt.close('scatter')
                
                plt.subplot(5,2,2)
                plt.axis('off')
                plt.plot(bx, by, '.', markersize=1, color = [0.0, 0.0, 0.0, 0.05])
                plt.plot(bx[AllSocialFrames_TF], by[AllSocialFrames_TF], '.', markersize=1, color = [1.0, 0.0, 0.0, 0.05], )
                plt.plot(bx[AllNONSocialFrames_TF], by[AllNONSocialFrames_TF], '.', markersize=1, color = [0.0, 0.0, 1.0, 0.05])
                plt.title('SPI: ' + format(SPI_s, '.3f') + ', VPI: ' + format(VPI_s, '.3f'))
                plt.axis([x_min, x_max, y_min, y_max])
                plt.gca().invert_yaxis()

                plt.subplot(5,2,4)
                plt.title('BPS: ' + format(BPS_s, '.3f') + ', %Paused: ' + format(Percent_Paused_s, '.2f') + ', %Moving: ' + format(Percent_Moving_s, '.2f'))
                motion[motion == -1.0] = -0.01
                plt.axhline(motionStartThreshold, c="green")
                plt.axhline(motionStopThreshold, c="red")
                plt.plot(motion)

                plt.subplot(5,2,6)
                motion[motion == -1.0] = -0.01
                plt.axhline(motionStartThreshold, c="green")
                plt.axhline(motionStopThreshold, c="red")
                plt.plot(motion[50000:51000])

                plt.subplot(5,2,8)
                plt.plot(avgBout_s, 'k')
                
                plt.subplot(5,2,10)
                plt.plot(area, 'r')
                
            #-----------------------------------
            # Save figure and data for each fish
            if plot:
                
                plt.savefig(plotFilename, dpi=600)
                plt.close('all')

            #----------------------------
            # Save Analyzed Summary Data
            
            np.savez(dataFilename,
                     VPI_NS=VPI_ns,
                     VPI_NS_BINS=VPI_ns_bins,
                     VPI_S=VPI_s,
                     VPI_S_BINS=VPI_s_bins,
                     SPI_NS=SPI_ns, 
                     SPI_S=SPI_s,
                     BPS_NS=BPS_ns,
                     BPS_S=BPS_s,
                     Distance_NS = Distance_ns,
                     Distance_S = Distance_s,
                     OrtHist_NS_NonSocialSide = OrtHist_ns_NonSocialSide,
                     OrtHist_NS_SocialSide = OrtHist_ns_SocialSide,
                     OrtHist_S_NonSocialSide = OrtHist_s_NonSocialSide,
                     OrtHist_S_SocialSide = OrtHist_s_SocialSide,
                     Bouts_NS = Bouts_ns, 
                     Bouts_S = Bouts_s,
                     Pauses_NS = Pauses_ns,
                     Pauses_S = Pauses_s,
                     Percent_Moving_NS = Percent_Moving_ns,
                     Percent_Moving_S = Percent_Moving_s,
                     Percent_Paused_NS = Percent_Paused_ns,
                     Percent_Paused_S = Percent_Paused_s,
                     Freezes_NS = int(Freezes_ns),
                     Freezes_S = int(Freezes_s),
                     Long_Freezes_NS = float(Long_Freezes_ns),
                     Long_Freezes_S = float(Long_Freezes_s),
                     # TR added 20/04/23
                     midCrossings_NS = int(midCrossings_ns),
                     midCrossings_S = int(midCrossings_s),
                     boutsAngles_NS = boutAngles_ns,
                     boutsAngles_S = boutAngles_s,
                     boutsDist_NS = boutDists_ns,
                     boutsDist_S = boutDists_s,
                     Freezes_X_NS = Freezes_X_ns,
                     Freezes_Y_NS = Freezes_Y_ns,
                     Freezes_X_S = Freezes_X_s,
                     Freezes_Y_S = Freezes_Y_s)
        else:
            print ("Bad Fish")
    
    # Report Porgress
    print (idx)
# End of Analysis Loop

#FIN
