# -*- coding: utf-8 -*-
"""
Create summary (figures and report) for all analyzed fish in a social preference experiment

@author: kampff
"""
#%% -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
#lib_path = r'/home/kampff/Repos/Dreosti-Lab/Social_Zebrafish/libs'
lib_path = r'S:\WIBR_Dreosti_Lab\Tom\Github\Lonely_Fish_TR\Libraries'
TR_lib_path = r'S:\WIBR_Dreosti_Lab\Tom\Github\Arena_Zebrafish\libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)
sys.path.append(TR_lib_path)
#-----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import seaborn as sns
import cv2

# Import local modules
import AZ_utilities as AZU
import SZ_utilities as SZU

save=True
keep=True
# freeze_threshold_seconds=4
# long_freeze_threshold_seconds=60
# gene = 'trio'
#base_path=r'S:/WIBR_Dreosti_Lab/Tom/Data/Lesion_Social/C-Chamber/Analysis'
# base_path=r'S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/AnalysisRounds'
base_path = r'D:\dataToTrack'
analysisRoot = base_path + r'/Analysis_Summer23/' 
# folderListFile = r'S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/FolderLists/testyD.txt'
# folderListFile = r'S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/FolderLists/cumulative_' + gene + '_cohort.txt' #folderListFile = 'S:/WIBR_Dreosti_Lab/Tom/Data/Lesion_Social/ShamCChamber.txt'

#%%##### Functions #############
def getNormHist(ns,s,bins,ranger):
    #Make histogram and plot it with lines 
    a_ns,c=np.histogram(ns,  bins=bins, range=ranger)
    a_s,c=np.histogram(s,  bins=bins, range=ranger)
    centers = (c[:-1]+c[1:])/2

    #Normalize by tot number of fish
    Tot_Fish_NS=len(ns)

    a_ns_float = np.float32(a_ns)
    a_s_float = np.float32(a_s)

    a_ns_nor_medium=a_ns_float/Tot_Fish_NS
    a_s_nor_medium=a_s_float/Tot_Fish_NS 
 
    return centers,a_ns_nor_medium,a_s_nor_medium

# def plotBoutAngleVsDist(centers,a_ns_nor_medium,a_s_nor_medium,filename, title,save=True,keep=True):

def cohortBoolGen(cohortList,cohortNum):
    ii=[]
    for i in cohortList:
        if i == cohortNum:
            ii.append(True)
        else:
            ii.append(False)
    return ii
    
def plotSummaryVPI(centers,a_ns_nor_medium,a_s_nor_medium,filename, title,save=True,keep=True):
    plt.figure()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplot(1,3,1)
    plt.plot(centers, a_ns_nor_medium, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
    plt.plot(centers, a_s_nor_medium, color=[1.0,0.0,0.0,0.5], linewidth=4.0)
    plt.title('Non Social/Social VPI', fontsize=12)
    plt.xlabel('Preference Index', fontsize=12)
    plt.ylabel('Rel. Frequency', fontsize=12)
    plt.axis([-1.1, 1.1, 0, 0.5])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
    plt.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
    
    bar_width=0.25
    plt.subplot(1,3,2)
    plt.bar(centers, a_ns_nor_medium, width=bar_width, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
    plt.title('Non Social VPI', fontsize=12)
    plt.xlabel('Preference Index', fontsize=12)
    plt.ylabel('Rel. Frequency', fontsize=12)
    plt.axis([-1.1, 1.1, 0, 0.5])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
    plt.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
    
    plt.subplot(1,3,3)
    plt.bar(centers, a_s_nor_medium, width=0.25, color=[1.0,0.0,0.0,1.0], linewidth=4.0)
    plt.title('Social VPI', fontsize=12)
    plt.xlabel('Preference Index', fontsize=12)
    plt.ylabel('Rel. Frequency', fontsize=12)
    plt.axis([-1.1, 1.1, 0, 0.5])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
    plt.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=12)  
    
    plt.suptitle(title)
    if save:
        plt.savefig(filename,dpi=600)
    if keep==False:
        plt.close()
        axes=-1
    else:
        axes=plt.gca
    return axes

def plotSummaryBinnedVPI(VPI_NS_BINS_ALL,VPI_S_BINS_ALL,filename,title,save=True,keep=False):
    plt.figure()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.title("Temporal VPI (one minute bins)")
    plt.subplot(1,2,1)
    m = np.nanmean(VPI_NS_BINS_ALL, 0)
    std = np.nanstd(VPI_NS_BINS_ALL, 0)
    valid = (np.logical_not(np.isnan(VPI_NS_BINS_ALL)))
    n = np.sum(valid, 0)
    se = std/np.sqrt(n)
    plt.plot(VPI_NS_BINS_ALL.T, linewidth=1, color=[0,0,0,0.1])
    plt.plot(m, 'k', linewidth=2)
    plt.plot(m, 'ko', markersize=5)
    plt.plot(m+se, 'r', linewidth=1)
    plt.plot(m-se, 'r', linewidth=1)
    plt.axis([0, 15, -1.1, 1.1])
    plt.xlabel('minutes')
    plt.ylabel('VPI')

    plt.subplot(1,2,2)
    m = np.nanmean(VPI_S_BINS_ALL, 0)
    std = np.nanstd(VPI_S_BINS_ALL, 0)
    valid = (np.logical_not(np.isnan(VPI_S_BINS_ALL)))
    n = np.sum(valid, 0)
    se = std/np.sqrt(n)
    plt.plot(VPI_S_BINS_ALL.T, linewidth=1, color=[0,0,0,0.1])
    plt.plot(m, 'k', linewidth=2)
    plt.plot(m, 'ko', markersize=5)
    plt.plot(m+se, 'r', linewidth=1)
    plt.plot(m-se, 'r', linewidth=1)
    plt.axis([0, 15, -1.1, 1.1])
    plt.xlabel('minutes')
    
    plt.suptitle(title)
    if save:
        plt.savefig(filename,dpi=600)
    if keep==False:
        plt.close()
        axes=-1
    else:
        axes=plt.gca
    return axes

# BPS Summary Plot
def plotSummaryBPS(centers,a_ns_nor_medium,a_s_nor_medium,filename, title,save=False,keep=True):
    plt.figure()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplot(1,3,1)
    plt.plot(centers, a_ns_nor_medium, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
    plt.plot(centers, a_s_nor_medium, color=[1.0,0.0,0.0,0.5], linewidth=4.0)
    plt.title('Non Social/Social BPS', fontsize=12)
    plt.xlabel('Bouts per Second (BPS)', fontsize=12)
    plt.ylabel('Rel. Frequency', fontsize=12)
    plt.axis([-0.1, 10.1, 0, 0.5])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
    plt.xticks([0, 5, 10], fontsize=12)
    
    bar_width=0.5
    plt.subplot(1,3,2)
    plt.bar(centers, a_ns_nor_medium, width=bar_width, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
    plt.title('Non Social BPS', fontsize=12)
    plt.xlabel('Bouts per Second (BPS)', fontsize=12)
    plt.ylabel('Rel. Frequency', fontsize=12)
    plt.axis([-0.1, 10.1, 0, 0.5])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
    plt.xticks([0, 5, 10], fontsize=12)
    
    plt.subplot(1,3,3)
    plt.bar(centers, a_s_nor_medium, width=bar_width, color=[1.0,0.0,0.0,1.0], linewidth=4.0)
    plt.title('Social BPS', fontsize=12)
    plt.xlabel('Bouts per Second (BPS)', fontsize=12)
    plt.ylabel('Rel. Frequency', fontsize=12)
    plt.axis([-0.1, 10.1, 0, 0.5])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=12)
    plt.xticks([0, 5, 10], fontsize=12)
    plt.suptitle(title)
    if save:
        plt.savefig(filename,dpi=600)
    if keep==False:
        plt.close()
        axes=-1
    else:
        axes=plt.gca
    return axes

def plotSummaryBouts(ns,s,filename,title,save=True,keep=False):
    plt.figure()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    bar_width=0.005
    
    # NS
    visible_bouts = np.where(ns[:,9] == 1)[0]
    non_visible_bouts = np.where(ns[:,9] == 0)[0]
    plt.subplot(221)
    bout_durations_ns, c = np.histogram(ns[non_visible_bouts,8], bins=51, range=(0,50))
    centers = (c[:-1]+c[1:])/2
    plt.bar(centers/100, bout_durations_ns, width=bar_width, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
    plt.title('Non Social Bout Durations', fontsize=12)
    plt.ylabel('Rel. Frequency', fontsize=12)
    plt.subplot(223)
    bout_durations_ns, c = np.histogram(ns[visible_bouts,8], bins=51, range=(0,50))
    centers = (c[:-1]+c[1:])/2
    plt.bar(centers/100, bout_durations_ns, width=bar_width, color=[0.5,0.5,0.5,1.0], linewidth=4.0)
    plt.title('Non Social Bout Durations', fontsize=12)
    plt.xlabel('Bout Durations (sec)', fontsize=12)
    plt.ylabel('Rel. Frequency', fontsize=12)
    # S
    visible_bouts = np.where(s[:,9] == 1)[0]
    non_visible_bouts = np.where(s[:,9] == 0)[0]
    plt.subplot(222)
    bout_durations_s, c = np.histogram(s[non_visible_bouts,8], bins=51, range=(0,50))
    centers = (c[:-1]+c[1:])/2
    plt.bar(centers/100, bout_durations_s, width=bar_width, color=[1.0,0.5,0.5,1.0], linewidth=4.0)
    plt.title('Social Bout Durations', fontsize=12)
    plt.ylabel('Rel. Frequency', fontsize=12)
    plt.subplot(224)
    bout_durations_s, c = np.histogram(s[visible_bouts,8], bins=51, range=(0,50))
    centers = (c[:-1]+c[1:])/2
    plt.bar(centers/100, bout_durations_s, width=bar_width, color=[1.0,0.5,0.5,1.0], linewidth=4.0)
    plt.title('Social Bout Durations', fontsize=12)
    plt.xlabel('Bout Durations (sec)', fontsize=12)
    plt.ylabel('Rel. Frequency', fontsize=12)
    plt.suptitle(title)
    
    if save:
        plt.savefig(filename,dpi=600)
    if keep==False:
        plt.close()
        axes=-1
    else:
        axes=plt.gca
    return axes

    # -------------------------------------------------------------------------
def plotSummaryBoutLocations(ns,s,filename,title,save=True,keep=False):
    # All Bouts Summary Plot
    plt.figure()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplot(1,2,1)
    plt.title('NonSocial Phase')
    plt.plot(ns[:, 1], ns[:, 2], '.', color=[0.0, 0.0, 0.0, 0.002])
    plt.axis([0, 17, 0, 42])
    plt.gca().invert_yaxis()
        
    plt.subplot(1,2,2)
    plt.title('Social Phase')
    plt.plot(s[:, 1], s[:, 2], '.', color=[0.0, 0.0, 0.0, 0.002])
    plt.axis([0, 17, 0, 42])
    plt.gca().invert_yaxis()
    plt.suptitle(title)
    
    if save:
        plt.savefig(filename,dpi=600)
    if keep==False:
        plt.close()
        axes=-1
    else:
        axes=plt.gca
    return axes

def plotSummaryVPIComparisons(VPIs,compMets_NS,compMets_S,f_prefix,t_suffix,save=True,keep=False):
    [VPI_ns,VPI_s]=VPIs
    [BPS_ns,Distance_ns,Freezes_ns]=compMets_NS
    [BPS_s,Distance_s,Freezes_s]=compMets_S
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.suptitle('VPI vs BPS ' + t_suffix )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.title('NonSocial Phase')
    plt.plot(VPI_ns, BPS_ns, 'o', color=[0.0, 0.0, 0.0, 0.5])
    plt.xlabel('Visual Preference Index (VPI)', fontsize=12)
    plt.ylabel('Bouts per Second', fontsize=12)
    plt.axis([-1.1, 1.1, 0, max(BPS_ns)+0.5])
    plt.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
    plt.ylim((0,6))
    
    plt.subplot(1,2,2)
    plt.title('Social Phase')
    plt.plot(VPI_s, BPS_s, 'o', color=[0.0, 0.0, 0.0, 0.5])
    plt.xlabel('Visual Preference Index (VPI)', fontsize=12)
    plt.axis([-1.1, 1.1, 0,  max(BPS_s)+0.5])
    plt.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
    plt.ylim((0,6))
    
    if save:
        nam='VPI_vs_BPS_'+t_suffix
        filename=f_prefix+nam+'.png'
        plt.savefig(filename,dpi=600)
    if keep==False:
        plt.close()
    # ----------------
    # VPI vs Distance Traveled Summary Plot 
    plt.figure()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplot(1,2,1)
    plt.suptitle('VPI vs Distance Traveled'+t_suffix)
    plt.title('NonSocial Phase')
    plt.plot(VPI_ns, Distance_ns, 'o', color=[0.0, 0.0, 0.0, 0.5])
    plt.xlabel('Visual Preference Index (VPI)', fontsize=12)
    plt.ylabel('Distance (mm)', fontsize=12)
    plt.axis([-1.1, 1.1, 0, max(Distance_ns)+0.5])
    plt.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
    plt.ylim((0,10000))
    
    plt.subplot(1,2,2)
    plt.title('Social Phase')
    plt.plot(VPI_s, Distance_s, 'o', color=[0.0, 0.0, 0.0, 0.5])
    plt.xlabel('Visual Preference Index (VPI)', fontsize=12)
    plt.axis([-1.1, 1.1, 0,  max(Distance_s)+0.5])
    plt.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
    plt.ylim((0,10000))
    
    if save:
        nam='VPI_vs_Dist'+t_suffix
        filename=f_prefix+nam+'.png'
        plt.savefig(filename,dpi=600)
    if keep==False:
        plt.close()
    # ----------------
    # VPI vs Number of Long Pauses Summary Plot 
    plt.figure()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplot(1,2,1)
    plt.suptitle('VPI vs Freezes'+t_suffix)
    plt.title('NonSocial Phase')
    plt.plot(VPI_ns, Freezes_ns, 'o', color=[0.0, 0.0, 0.0, 0.5])
    plt.xlabel('Visual Preference Index (VPI)', fontsize=12)
    plt.ylabel('Freezes (count)', fontsize=12)
    plt.axis([-1.1, 1.1, -1, max(Freezes_ns)+0.5])
    plt.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
    plt.ylim((0,40))
    
    plt.subplot(1,2,2)
    plt.title('Social Phase')
    plt.plot(VPI_s, Freezes_s, 'o', color=[0.0, 0.0, 0.0, 0.5])
    plt.xlabel('Visual Preference Index (VPI)', fontsize=12)
    plt.axis([-1.1, 1.1, -1, max(Freezes_s)+0.5])
    plt.xticks([-1, -0.5, 0, 0.5, 1.0], fontsize=12)
    plt.ylim((0,40))
    
    if save:
        nam='VPI_vs_Freezes'+t_suffix
        filename=f_prefix+nam+'.png'
        plt.savefig(filename,dpi=600)
    if keep==False:
        plt.close()
    return

def plotSummaryOrtHistograms(OrtHist_NS_NSS_ALL,OrtHist_NS_SS_ALL,OrtHist_S_NSS_ALL,OrtHist_S_SS_ALL, title, filename='',save=True,keep=False):
    # Accumulate all histogram values and normalize
    Accum_OrtHist_NS_NSS_ALL = np.sum(OrtHist_NS_NSS_ALL, axis=0)
    Accum_OrtHist_NS_SS_ALL = np.sum(OrtHist_NS_SS_ALL, axis=0)
    Accum_OrtHist_S_NSS_ALL = np.sum(OrtHist_S_NSS_ALL, axis=0)
    Accum_OrtHist_S_SS_ALL= np.sum(OrtHist_S_SS_ALL, axis=0)
    
    Norm_OrtHist_NS_NSS_ALL = Accum_OrtHist_NS_NSS_ALL/np.sum(Accum_OrtHist_NS_NSS_ALL)
    Norm_OrtHist_NS_SS_ALL = Accum_OrtHist_NS_SS_ALL/np.sum(Accum_OrtHist_NS_SS_ALL)
    Norm_OrtHist_S_NSS_ALL = Accum_OrtHist_S_NSS_ALL/np.sum(Accum_OrtHist_S_NSS_ALL)
    Norm_OrtHist_S_SS_ALL = Accum_OrtHist_S_SS_ALL/np.sum(Accum_OrtHist_S_SS_ALL)
    
    # Plot Summary
    xAxis = np.arange(-np.pi,np.pi+np.pi/18.0, np.pi/18.0)
    #plt.figure('Summary: Orientation Histograms')
    plt.figure()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(title)
    ax1 = plt.subplot(221, polar=True)
    plt.title('NS - Non Social Side')
    plt.plot(xAxis, np.hstack((Norm_OrtHist_NS_NSS_ALL, Norm_OrtHist_NS_NSS_ALL[0])), linewidth = 3)
    
    ax2 = plt.subplot(222, polar=True)
    plt.title('NS - Social Side')
    plt.plot(xAxis, np.hstack((Norm_OrtHist_NS_SS_ALL, Norm_OrtHist_NS_SS_ALL[0])), linewidth = 3)
    
    ax3 = plt.subplot(223, polar=True)
    plt.title('S - Non Social Side')
    plt.plot(xAxis, np.hstack((Norm_OrtHist_S_NSS_ALL, Norm_OrtHist_S_NSS_ALL[0])), linewidth = 3)
    
    ax4 = plt.subplot(224, polar=True)
    plt.title('S - Social Side')
    plt.plot(xAxis, np.hstack((Norm_OrtHist_S_SS_ALL, Norm_OrtHist_S_SS_ALL[0])), linewidth = 3)

    if save:
        plt.savefig(filename,dpi=600)
    if keep:
        return ax1,ax2,ax3,ax4
    else:
        return -1,-1,-1,-1

#%%######### SCRIPT #############
groups, num, folderNames, fishStatus, ROI_path = SZU.read_folder_list(folderListFile)

# Create a list of unique genes from the folderListFile, to analyse in sequence
FPSs=[]
geneList=[]
for idx,folder in enumerate(folderNames):
    
    gene=folder.rsplit(sep='\\',maxsplit=3)[1]
    if gene not in geneList:
        geneList.append(gene)
    
    # Get Folder Names
    NS_folder, S_folder, _ = SZU.get_folder_names(folder)

    # Determine FPS for this fish set
    aviFiles = glob.glob(NS_folder+'/*.avi')
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    currentFPS=vid.get(cv2.CAP_PROP_FPS)
    vid.release()
    
    stat=fishStatus[idx]
    for j in range(0,6):
        if stat[j]==1:
            FPSs.append(currentFPS)

# loop through tested genes
for gene in geneList:
    print('Summarizing group : ' + str(len(FPSs)) + ' ' + gene + ' fish')
    analysisFolder = analysisRoot + gene
    AZU.cycleMkDir_forw(analysisFolder)
    # Find all the npz files saved for each group and fish with all the information
    npzFiles = glob.glob(analysisFolder+'/*.npz')
    analysisFolder=analysisFolder+'/Summary'
    AZU.cycleMkDir_forw(analysisFolder)
    # find how many are in each cohort
    cohortList=[]
    for f, filename in enumerate(npzFiles):
        cohortList.append(int(filename.rsplit(sep='\\')[1][0]))
    num_cohorts=len(set(cohortList))

    # Calculate how many files
    numFiles_ALL = np.size(npzFiles, 0)

    # Allocate space for summary data_ALL
    VPI_NS_ALL = np.zeros(numFiles_ALL)
    VPI_S_ALL = np.zeros(numFiles_ALL)
    VPI_NS_BINS_ALL = np.zeros((numFiles_ALL, 15))
    VPI_S_BINS_ALL = np.zeros((numFiles_ALL, 15))
    SPI_NS_ALL = np.zeros(numFiles_ALL)
    SPI_S_ALL = np.zeros(numFiles_ALL)
    BPS_NS_ALL = np.zeros(numFiles_ALL)
    BPS_S_ALL = np.zeros(numFiles_ALL)
    Distance_NS_ALL = np.zeros(numFiles_ALL)
    Distance_S_ALL = np.zeros(numFiles_ALL)
    Freezes_NS_ALL = np.zeros(numFiles_ALL)
    Freezes_S_ALL = np.zeros(numFiles_ALL)
    Long_Freezes_NS_ALL = np.zeros(numFiles_ALL)
    Long_Freezes_S_ALL = np.zeros(numFiles_ALL)
    Percent_Moving_NS_ALL = np.zeros(numFiles_ALL)
    Percent_Moving_S_ALL = np.zeros(numFiles_ALL)
    OrtHist_NS_NSS_ALL = np.zeros((numFiles_ALL,36))
    OrtHist_NS_SS_ALL = np.zeros((numFiles_ALL,36))
    OrtHist_S_NSS_ALL = np.zeros((numFiles_ALL,36))
    OrtHist_S_SS_ALL = np.zeros((numFiles_ALL,36))
    Bouts_NS_ALL = np.zeros((0,10))
    Bouts_S_ALL = np.zeros((0,10))
    Pauses_NS_ALL = np.zeros((0,10))   
    Pauses_S_ALL = np.zeros((0,10))
    
    cohortBoolean=[]
    
    # Create report file ALL
    reportFilename_ALL = analysisFolder + r'/' + gene + '_report_ALL.txt'
    reportFile_ALL = open(reportFilename_ALL, 'w')

    for cohortNum in range(1,num_cohorts+1):
        cohortBoolean.append(cohortBoolGen(cohortList,cohortNum))
    
    # Create Lists to store all cohorts
    VPI_NS_COHORT_list = []
    VPI_S_COHORT_list = []
    VPI_NS_BINS_COHORT_list = []
    VPI_S_BINS_COHORT_list = []
    SPI_NS_COHORT_list = []
    SPI_S_COHORT_list = []
    BPS_NS_COHORT_list = []
    BPS_S_COHORT_list = []
    Distance_NS_COHORT_list = []
    Distance_S_COHORT_list = []
    Freezes_NS_COHORT_list = []
    Freezes_S_COHORT_list = []
    Long_Freezes_NS_COHORT_list = []
    Long_Freezes_S_COHORT_list = []
    Percent_Moving_NS_COHORT_list = []
    Percent_Moving_S_COHORT_list = []
    OrtHist_NS_NSS_COHORT_list = []
    OrtHist_NS_SS_COHORT_list = []
    OrtHist_S_NSS_COHORT_list = []
    OrtHist_S_SS_COHORT_list = []
    Bouts_NS_COHORT_list = []
    Bouts_S_COHORT_list = []
    Pauses_NS_COHORT_list = []
    Pauses_S_COHORT_list = []
    
    ##### START OF COHORT LOOP
    ALL_fileCount=0
    for cohortIdx,cohortBool in enumerate(cohortBoolean):
        cohortNum=cohortIdx+1
        # Calculate how many files this cohort
        numFiles_COHORT = np.sum(cohortBoolean[cohortIdx])
        
        # Create report file COHORT
        reportFilename_COHORT = analysisFolder + r'/' + gene + '_report_COHORT_' + str(cohortNum) + '.txt'
        reportFile_COHORT = open(reportFilename_COHORT, 'w')
        
        # Make arrays to store COHORT
        VPI_NS_COHORT = np.zeros(numFiles_COHORT)
        VPI_S_COHORT = np.zeros(numFiles_COHORT)
        VPI_NS_BINS_COHORT = np.zeros((numFiles_COHORT, 15))
        VPI_S_BINS_COHORT = np.zeros((numFiles_COHORT, 15))
        SPI_NS_COHORT = np.zeros(numFiles_COHORT)
        SPI_S_COHORT = np.zeros(numFiles_COHORT)
        BPS_NS_COHORT = np.zeros(numFiles_COHORT)
        BPS_S_COHORT = np.zeros(numFiles_COHORT)
        Distance_NS_COHORT = np.zeros(numFiles_COHORT)
        Distance_S_COHORT = np.zeros(numFiles_COHORT)
        Freezes_NS_COHORT = np.zeros(numFiles_COHORT)
        Freezes_S_COHORT = np.zeros(numFiles_COHORT)
        Long_Freezes_NS_COHORT = np.zeros(numFiles_COHORT)
        Long_Freezes_S_COHORT = np.zeros(numFiles_COHORT)
        Percent_Moving_NS_COHORT = np.zeros(numFiles_COHORT)
        Percent_Moving_S_COHORT = np.zeros(numFiles_COHORT)
        OrtHist_NS_NSS_COHORT = np.zeros((numFiles_COHORT,36))
        OrtHist_NS_SS_COHORT = np.zeros((numFiles_COHORT,36))
        OrtHist_S_NSS_COHORT = np.zeros((numFiles_COHORT,36))
        OrtHist_S_SS_COHORT = np.zeros((numFiles_COHORT,36))
        Bouts_NS_COHORT = np.zeros((0,10))
        Bouts_S_COHORT = np.zeros((0,10))
        Pauses_NS_COHORT = np.zeros((0,10))   
        Pauses_S_COHORT = np.zeros((0,10))
        
        # Find all indexes for this cohort
        indexes=np.where(cohortBool)[0] 
    
        # Go through all the files contained in the analysis folder FOR THIS COHORT
        ##### Start of File Loop
        for fidx,f in enumerate(indexes):
            filename=npzFiles[f]
            # Load each npz file
            dataobject = np.load(filename)
            
            # Extract from the npz file
            VPI_NS = dataobject['VPI_NS']    
            VPI_S = dataobject['VPI_S']   
            VPI_NS_BINS = dataobject['VPI_NS_BINS']    
            VPI_S_BINS = dataobject['VPI_S_BINS']
            SPI_NS = dataobject['SPI_NS']    
            SPI_S = dataobject['SPI_S']   
            BPS_NS = dataobject['BPS_NS']   
            BPS_S = dataobject['BPS_S']
            Distance_NS = dataobject['Distance_NS']   
            Distance_S = dataobject['Distance_S']   
            OrtHist_ns_NonSocialSide = dataobject['OrtHist_NS_NonSocialSide']
            OrtHist_ns_SocialSide = dataobject['OrtHist_NS_SocialSide']
            OrtHist_s_NonSocialSide = dataobject['OrtHist_S_NonSocialSide']
            OrtHist_s_SocialSide = dataobject['OrtHist_S_SocialSide']
            Bouts_NS = dataobject['Bouts_NS']   
            Bouts_S = dataobject['Bouts_S']
            Pauses_NS = dataobject['Pauses_NS']   
            Pauses_S = dataobject['Pauses_S']
            Percent_Moving_NS = dataobject['Percent_Moving_NS']   
            Percent_Moving_S = dataobject['Percent_Moving_S']
            Freezes_NS = dataobject['Freezes_NS']
            Freezes_S = dataobject['Freezes_S']
            Long_Freezes_NS = dataobject['Long_Freezes_NS']
            Long_Freezes_S = dataobject['Long_Freezes_S']
            
            # Store COHORT summary stats
            VPI_NS_COHORT[fidx] = VPI_NS
            VPI_S_COHORT[fidx] = VPI_S
            VPI_NS_BINS_COHORT[fidx,:] = VPI_NS_BINS
            VPI_S_BINS_COHORT[fidx,:] = VPI_S_BINS
            SPI_NS_COHORT[fidx] = SPI_NS
            SPI_S_COHORT[fidx] = SPI_S
            BPS_NS_COHORT[fidx] = BPS_NS
            BPS_S_COHORT[fidx] = BPS_S
            Distance_NS_COHORT[fidx] = Distance_NS
            Distance_S_COHORT[fidx] = Distance_S
            Freezes_S_COHORT[fidx] = Freezes_S
            Freezes_NS_COHORT[fidx] = Freezes_NS
            Long_Freezes_NS_COHORT[fidx] = Long_Freezes_NS
            Long_Freezes_S_COHORT[fidx] = Long_Freezes_S
            OrtHist_NS_NSS_COHORT[fidx,:] = OrtHist_ns_NonSocialSide
            OrtHist_NS_SS_COHORT[fidx,:] = OrtHist_ns_SocialSide
            OrtHist_S_NSS_COHORT[fidx,:] = OrtHist_s_NonSocialSide
            OrtHist_S_SS_COHORT[fidx,:] = OrtHist_s_SocialSide
        
            # Concat ALL Pauses/Bouts
            Bouts_NS_COHORT = np.vstack([Bouts_NS_COHORT, Bouts_NS])
            Bouts_S_COHORT = np.vstack([Bouts_S_COHORT, Bouts_S])
            Pauses_NS_COHORT = np.vstack([Pauses_NS_COHORT, Pauses_NS])
            Pauses_S_COHORT = np.vstack([Pauses_S_COHORT, Pauses_S])
            
            # Store ALL summary stats
            VPI_NS_ALL[ALL_fileCount] = VPI_NS
            VPI_S_ALL[ALL_fileCount] = VPI_S
            VPI_NS_BINS_ALL[ALL_fileCount,:] = VPI_NS_BINS
            VPI_S_BINS_ALL[ALL_fileCount,:] = VPI_S_BINS
            SPI_NS_ALL[ALL_fileCount] = SPI_NS
            SPI_S_ALL[ALL_fileCount] = SPI_S
            BPS_NS_ALL[ALL_fileCount] = BPS_NS
            BPS_S_ALL[ALL_fileCount] = BPS_S
            Distance_NS_ALL[ALL_fileCount] = Distance_NS
            Distance_S_ALL[ALL_fileCount] = Distance_S
            Freezes_S_ALL[ALL_fileCount] = Freezes_S
            Freezes_NS_ALL[ALL_fileCount] = Freezes_NS
            Long_Freezes_NS_ALL[ALL_fileCount] = Long_Freezes_NS
            Long_Freezes_S_ALL[ALL_fileCount] = Long_Freezes_S
            OrtHist_NS_NSS_ALL[ALL_fileCount,:] = OrtHist_ns_NonSocialSide
            OrtHist_NS_SS_ALL[ALL_fileCount,:] = OrtHist_ns_SocialSide
            OrtHist_S_NSS_ALL[ALL_fileCount,:] = OrtHist_s_NonSocialSide
            OrtHist_S_SS_ALL[ALL_fileCount,:] = OrtHist_s_SocialSide
        
            # Concat ALL Pauses/Bouts
            Bouts_NS_ALL = np.vstack([Bouts_NS_ALL, Bouts_NS])
            Bouts_S_ALL = np.vstack([Bouts_S_ALL, Bouts_S])
            Pauses_NS_ALL = np.vstack([Pauses_NS_ALL, Pauses_NS])
            Pauses_S_ALL = np.vstack([Pauses_S_ALL, Pauses_S])
            
            ALL_fileCount+=1
                
            # Save to COHORT report (text file)
            reportFile_COHORT.write(filename + '\n')
            reportFile_COHORT.write('-------------------\n')
            reportFile_COHORT.write('Cohort ' + str(cohortNum) + '\n')
            reportFile_COHORT.write('VPI_NS:\t' + format(np.float64(VPI_NS), '.3f') + '\n')
            reportFile_COHORT.write('VPI_S:\t' + format(np.float64(VPI_S), '.3f') + '\n')
            reportFile_COHORT.write('SPI_NS:\t' + format(np.float64(SPI_NS), '.3f') + '\n')
            reportFile_COHORT.write('SPI_S:\t' + format(np.float64(SPI_S), '.3f') + '\n')
            reportFile_COHORT.write('BPS_NS:\t' + format(np.float64(BPS_NS), '.3f') + '\n')
            reportFile_COHORT.write('BPS_S:\t' + format(np.float64(BPS_S), '.3f') + '\n')
            reportFile_COHORT.write('Distance_NS:\t' + format(np.float64(Distance_NS), '.3f') + '\n')
            reportFile_COHORT.write('Distance_S:\t' + format(np.float64(Distance_S), '.3f') + '\n')
            reportFile_COHORT.write('Freezes_NS:\t' + format(np.float64(Freezes_NS), '.3f') + '\n')
            reportFile_COHORT.write('Freezes_S:\t' + format(np.float64(Freezes_S), '.3f') + '\n')
            reportFile_COHORT.write('Long_Freezes_NS:\t' + format(np.float64(Long_Freezes_NS), '.3f') + '\n')
            reportFile_COHORT.write('Long_Freezes_S:\t' + format(np.float64(Long_Freezes_S), '.3f') + '\n')
            reportFile_COHORT.write('Perc_Moving_NS:\t' + format(np.float64(Percent_Moving_NS), '.3f') + '\n')
            reportFile_COHORT.write('Perc_Moving_S:\t' + format(np.float64(Percent_Moving_S), '.3f') + '\n')
            reportFile_COHORT.write('-------------------\n\n')
            
            # Save to ALL report (text file)
            reportFile_ALL.write(filename + '\n')
            reportFile_ALL.write('-------------------\n')
            reportFile_ALL.write('VPI_NS:\t' + format(np.float64(VPI_NS), '.3f') + '\n')
            reportFile_ALL.write('VPI_S:\t' + format(np.float64(VPI_S), '.3f') + '\n')
            reportFile_ALL.write('SPI_NS:\t' + format(np.float64(SPI_NS), '.3f') + '\n')
            reportFile_ALL.write('SPI_S:\t' + format(np.float64(SPI_S), '.3f') + '\n')
            reportFile_ALL.write('BPS_NS:\t' + format(np.float64(BPS_NS), '.3f') + '\n')
            reportFile_ALL.write('BPS_S:\t' + format(np.float64(BPS_S), '.3f') + '\n')
            reportFile_ALL.write('Distance_NS:\t' + format(np.float64(Distance_NS), '.3f') + '\n')
            reportFile_ALL.write('Distance_S:\t' + format(np.float64(Distance_S), '.3f') + '\n')
            reportFile_ALL.write('Freezes_NS:\t' + format(np.float64(Freezes_NS), '.3f') + '\n')
            reportFile_ALL.write('Freezes_S:\t' + format(np.float64(Freezes_S), '.3f') + '\n')
            reportFile_ALL.write('Long_Freezes_NS:\t' + format(np.float64(Long_Freezes_NS), '.3f') + '\n')
            reportFile_ALL.write('Long_Freezes_S:\t' + format(np.float64(Long_Freezes_S), '.3f') + '\n')
            reportFile_ALL.write('Perc_Moving_NS:\t' + format(np.float64(Percent_Moving_NS), '.3f') + '\n')
            reportFile_ALL.write('Perc_Moving_S:\t' + format(np.float64(Percent_Moving_S), '.3f') + '\n')
            reportFile_ALL.write('-------------------\n\n')
        
        ## END OF FILE LOOP
        # Close COHORT report
        reportFile_COHORT.close()
        
        # Store cohort in list
        VPI_NS_COHORT_list.append(VPI_NS_COHORT)
        VPI_S_COHORT_list.append(VPI_S_COHORT)
        VPI_NS_BINS_COHORT_list.append(VPI_NS_BINS_COHORT)
        VPI_S_BINS_COHORT_list.append(VPI_S_BINS_COHORT)
        SPI_NS_COHORT_list.append(SPI_NS_COHORT)
        SPI_S_COHORT_list.append(SPI_S_COHORT)
        BPS_NS_COHORT_list.append(BPS_NS_COHORT)
        BPS_S_COHORT_list.append(BPS_S_COHORT)
        Distance_NS_COHORT_list.append(Distance_NS_COHORT)
        Distance_S_COHORT_list.append(Distance_S_COHORT)
        Freezes_NS_COHORT_list.append(Freezes_NS_COHORT)
        Freezes_S_COHORT_list.append(Freezes_S_COHORT)
        Long_Freezes_NS_COHORT_list.append(Long_Freezes_NS_COHORT)
        Long_Freezes_S_COHORT_list.append(Long_Freezes_S_COHORT)
        Percent_Moving_NS_COHORT_list.append(Percent_Moving_NS_COHORT)
        Percent_Moving_S_COHORT_list.append(Percent_Moving_S_COHORT)
        OrtHist_NS_NSS_COHORT_list.append(OrtHist_NS_NSS_COHORT)
        OrtHist_NS_SS_COHORT_list.append(OrtHist_NS_SS_COHORT)
        OrtHist_S_NSS_COHORT_list.append(OrtHist_S_NSS_COHORT)
        OrtHist_S_SS_COHORT_list.append(OrtHist_S_SS_COHORT)
        Bouts_NS_COHORT_list.append(Bouts_NS_COHORT)
        Bouts_S_COHORT_list.append(Bouts_S_COHORT)
        Pauses_NS_COHORT_list.append(Pauses_NS_COHORT)
        Pauses_S_COHORT_list.append(Pauses_S_COHORT)
        # END OF COHORT LOOP
    
    # Close report
    reportFile_ALL.close()
#------------------------------------------------------------------------------
#%% Create a table with all behavioural metrics for each fish. Rows are fish, columns are:


    
# Genotype, VPI, Freezes, % time moving, Distance travelled, BPS... there will be more but this for now



    #%% Figures 
    # VPI Summary
    #------------------------------------------------------------------------------
    
    # VPI Summary Plot (ALL)
    centers,a_ns_nor_medium,a_s_nor_medium = getNormHist(VPI_NS_ALL,VPI_S_ALL,bins=8,ranger=(-1,1))
    filename = analysisFolder + '/' + gene + '_VPI_ALL.png'
    title='VPI Summary ALL'
    plotSummaryVPI(centers,a_ns_nor_medium,a_s_nor_medium,filename, title,save=save,keep=keep)
    
    # VPI Summary Plot (COHORT)
    for i in range(num_cohorts):
        centers,a_ns_nor_medium,a_s_nor_medium = getNormHist(VPI_NS_COHORT_list[i],VPI_S_COHORT_list[i],bins=8,ranger=(-1,1))
        filename = analysisFolder + '/' + gene + '_VPI_COHORT_' + str(i) + '.png'
        title='VPI Summary Cohort ' + str(i)
        plotSummaryVPI(centers,a_ns_nor_medium,a_s_nor_medium,filename, title,save=save,keep=keep)
        
    #------------------------------------------------------------------------------
    # VPI "Binned" Summary Plot
    #------------------------------------------------------------------------------
    # ALL
    filename = analysisFolder + '/' + gene + '_VPI_BINS_ALL.png'
    title="Temporal VPI (one minute bins) ALL"
    plotSummaryBinnedVPI(VPI_NS_BINS_ALL,VPI_S_BINS_ALL,filename,title,save=save,keep=keep)
    # COHORTS
    for i in range(num_cohorts):
        filename = analysisFolder + '/' + gene + '_VPI_BINS_COHORT_' + str(i) + '.png'
        title='Temporal VPI (one minute bins) COHORT ' + str(i)
        VPI_NS_BINS=VPI_NS_BINS_COHORT_list[i]
        VPI_S_BINS=VPI_S_BINS_COHORT_list[i]
        plotSummaryBinnedVPI(VPI_NS_BINS,VPI_S_BINS,filename,title,save=save,keep=keep)
    
    # SPI Summary Plots
    #------------------------------------------------------------------------------
    # SPI ALL
    filename = analysisFolder + '/' + gene + '_SPI_ALL.png'
    title='SPI Summary ALL'
    centers,a_ns_nor_medium,a_s_nor_medium=getNormHist(SPI_NS_ALL,SPI_S_ALL,bins=8,ranger=(-1,1))
    plotSummaryVPI(centers,a_ns_nor_medium,a_s_nor_medium,filename, title,save=save,keep=keep)
    
    # SPI Cohorts
    for i in range(num_cohorts):
        filename = analysisFolder + '/' + gene + '_SPI_COHORT_' + str(i) + '.png'
        title='SPI Summary COHORT ' + str(i)
        SPI_NS=SPI_NS_COHORT_list[i]
        SPI_S=SPI_S_COHORT_list[i]
        centers,a_ns_nor_medium,a_s_nor_medium=getNormHist(SPI_NS,SPI_S,bins=8,ranger=(-1,1))
        plotSummaryVPI(centers,a_ns_nor_medium,a_s_nor_medium,filename, title,save=save,keep=keep)
        
    #------------------------------------------------------------------------------
    # BPS ALL
    centers,a_ns_nor_medium,a_s_nor_medium=getNormHist(BPS_NS_ALL,BPS_S_ALL,bins=16,ranger=(0,10))
    # Make histogram and plot it with lines 
    title='BPS ALL'
    filename = analysisFolder + '/' + gene + '_BPS_ALL.png'
    plotSummaryBPS(centers,a_ns_nor_medium,a_s_nor_medium,filename, title,save=save,keep=keep)
    
    # BPS COHORTS
    for i in range(num_cohorts):
        filename = analysisFolder + '/' + gene + '_BPS_COHORT_' + str(i+1) + '.png'
        title='BPS Summary COHORT ' + str(i+1)
        BPS_NS=BPS_NS_COHORT_list[i]
        BPS_S=BPS_S_COHORT_list[i]
        centers,a_ns_nor_medium,a_s_nor_medium=getNormHist(BPS_NS,BPS_S,bins=16,ranger=(0,10))
        plotSummaryBPS(centers,a_ns_nor_medium,a_s_nor_medium,filename, title,save=save,keep=keep)
    
    # -----------------------------------------------------------------------------
    # Bouts Summary Plot
    # Bouts ALL
    title='Bouts ALL ' + str(gene)
    filename = analysisFolder + '/' + gene + '_Bouts_ALL.png'
    plotSummaryBouts(Bouts_NS_ALL,Bouts_S_ALL,filename,title,save=save,keep=keep)
    # Bouts Cohorts
    for i in range(num_cohorts):
        filename = analysisFolder + '/' + gene + '_Bouts_COHORT_' + str(i+1) + '.png'
        title='Bouts COHORT ' + str(i+1) + ' ' + str(gene)
        Bouts_NS=Bouts_NS_COHORT_list[i]
        Bouts_S=Bouts_S_COHORT_list[i]
        plotSummaryBouts(Bouts_NS,Bouts_S,filename,title,save=save,keep=keep)
        
    # Bout Locations
    filename = analysisFolder + '/' + gene + '_BoutLocations_ALL.png'
    title='Bout Location Summary ALL ' + str(gene)
    plotSummaryBoutLocations(Bouts_NS_ALL,Bouts_S_ALL,filename,title,save=save,keep=keep)
    # Bout Locations Cohorts
    for i in range(num_cohorts):
        filename = analysisFolder + '/' + gene + '_BoutLocations_COHORT_' + str(i+1) + '.png'
        title='Bout Location Summary COHORT ' + str(i+1) + ' ' + str(gene)
        Bouts_NS=Bouts_NS_COHORT_list[i]
        Bouts_S=Bouts_S_COHORT_list[i]
        plotSummaryBoutLocations(Bouts_NS,Bouts_S,filename,title,save=save,keep=keep)
    
        
    # ----------------
    # Long Pauses Summary Plot
    # plt.figure()
    # plt.subplot(1,2,1)
    # long_pauses_ns = np.where(Pauses_NS_ALL[:,8] > freeze_threshold)
    # num_long_pauses_per_fish_ns = len(long_pauses_ns)/numFiles
    # plt.title('NS: #Long Pauses = ' + format(num_long_pauses_per_fish_ns, '.4f'))
    # plt.plot(Pauses_NS_ALL[long_pauses_ns, 1], Pauses_NS_ALL[long_pauses_ns, 2], 'o', color=[0.0, 0.0, 0.0, 0.2])
    # plt.axis([0, 17, 0, 42])
    # plt.gca().invert_yaxis()
        
    # plt.subplot(1,2,2)
    # long_pauses_s = np.where(Pauses_S_ALL[:,8] > freeze_threshold)
    # num_long_pauses_per_fish_s = len(long_pauses_s)/numFiles
    # plt.title('S: #Long Pauses = ' + format(num_long_pauses_per_fish_s, '.4f'))
    # plt.plot(Pauses_S_ALL[long_pauses_s, 1], Pauses_S_ALL[long_pauses_s, 2], 'o', color=[0.0, 0.0, 0.0, 0.2])
    # plt.axis([0, 17, 0, 42])
    # plt.gca().invert_yaxis()
    
    # ----------------
    # VPI vs BPS Summary Plot 
    # VPI vs metrics ALL
    compMets_NS=[BPS_NS_ALL,Distance_NS_ALL,Freezes_NS_ALL]#Long_Freezes_NS_ALL]
    compMets_S=[BPS_S_ALL,Distance_S_ALL,Freezes_S_ALL]#Long_Freezes_S_ALL]
    VPIs=[VPI_NS_ALL,VPI_S_ALL]
    filename_prefix=analysisFolder + '/' + gene + '_'
    title_suffix = '_ALL'
    plotSummaryVPIComparisons(VPIs,compMets_NS,compMets_S,filename_prefix,title_suffix,save=save,keep=keep)
    
    # VPI vs metrics Cohorts
    for i in range(num_cohorts):
        filename_prefix=analysisFolder + '/' + gene
        title_suffix='_COHORT_' + str(i+1)
        VPIs=[VPI_NS_COHORT_list[i],VPI_S_COHORT_list[i]]
        compMets_NS=[BPS_NS_COHORT_list[i],Distance_NS_COHORT_list[i],Freezes_NS_COHORT_list[i]]
        compMets_S=[BPS_S_COHORT_list[i],Distance_S_COHORT_list[i],Freezes_S_COHORT_list[i]]#Long_Freezes_S_ALL]
        plotSummaryVPIComparisons(VPIs,compMets_NS,compMets_S,filename_prefix,title_suffix,save=save,keep=keep)
    
    # ----------------
    # ORT_HIST Summary Plot
    # Heading histograms ALL    
    title='Heading Summary ALL' + str(gene)
    filename = analysisFolder + '/' + gene + '_OrtHist_ALL.png'
    plotSummaryOrtHistograms(OrtHist_NS_NSS_ALL,OrtHist_NS_SS_ALL,OrtHist_S_NSS_ALL,OrtHist_S_SS_ALL, title, filename=filename,save=save,keep=keep)
    
    # Heading histograms Cohorts
    for i in range(num_cohorts):
        filename = analysisFolder + '/' + gene + '_OrtHist_COHORT_' + str(i+1) + '.png'
        title='Bout Location Summary COHORT ' + str(i+1) + ' ' + str(gene)
        OrtHist_NS_NSS = OrtHist_NS_NSS_COHORT_list[i]
        OrtHist_NS_SS = OrtHist_NS_SS_COHORT_list[i]
        OrtHist_S_NSS = OrtHist_S_NSS_COHORT_list[i]
        OrtHist_S_SS = OrtHist_S_SS_COHORT_list[i]
        plotSummaryOrtHistograms(OrtHist_NS_NSS,OrtHist_NS_SS,OrtHist_S_NSS,OrtHist_S_SS, title,filename=filename,save=save,keep=keep)
    
    #------------------------
    # Behaviour Summary plots
    # Create a list of Cohort numbers so you can color the points
    # VPI
    # colors=[[0.75,0,0,0.6],[0,0,0.75,0.6],[0,0.75,0,0.6],[0.75,0.75,0,0.6]]
    num_colors = 8

    # Create an array of evenly spaced values between 0 and 1
    values = np.linspace(0, 1, num_colors)
    
    # Use matplotlib's colormap to convert the values to RGBA colors
    colors = plt.cm.Spectral(values)
    
    # Convert the colors to 8-bit RGBA
    # colors = [(int(r*255), int(g*255), int(b*255), int(a*255)) for r,g,b,a in colors]

    plt.figure()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplot(2,2,1)
    plt.title('VPI')
    
    # plot the spots for all cohorts
    for i in range(num_cohorts):
        s1 = pd.Series(VPI_NS_COHORT_list[i], name='NS')
        s2 = pd.Series(VPI_S_COHORT_list[i], name='S')
        df = pd.concat([s1,s2], axis=1)
        # sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray", color=colors[i])
        ax=sns.violinplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray", color=colors[i])
        plt.setp(ax.collections, alpha=.3)
        # plot the bar for all fish
    s1 = pd.Series(VPI_NS_ALL, name='NS')
    s2 = pd.Series(VPI_S_ALL, name='S')
    df = pd.concat([s1,s2], axis=1)
    # sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], ci=95, capsize=0.05, errwidth=2)
    
    # BPS
    plt.subplot(2,2,2)
    plt.title('BPS')
    
    # plot the spots for all cohorts
    for i in range(num_cohorts):
        s1 = pd.Series(BPS_NS_COHORT_list[i], name='NS')
        s2 = pd.Series(BPS_S_COHORT_list[i], name='S')
        df = pd.concat([s1,s2], axis=1)
        # sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray", color=colors[i])
        sns.violinplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray", color=colors[i])
        
        # plot the bar for all fish
    # sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], ci=95, capsize=0.05, errwidth=2)
    s1 = pd.Series(BPS_NS_ALL, name='NS')
    s2 = pd.Series(BPS_S_ALL, name='S')
    df = pd.concat([s1,s2], axis=1)
    # sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], ci=95, capsize=0.05, errwidth=2)
    
    # Distance
    plt.subplot(2,2,3)
    plt.title('Distance Traveled')
    for i in range(num_cohorts):
        s1 = pd.Series(Distance_NS_COHORT_list[i], name='NS')
        s2 = pd.Series(Distance_S_COHORT_list[i], name='S')
        df = pd.concat([s1,s2], axis=1)
        # sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray", color=colors[i])
        sns.violinplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray", color=colors[i])
    
    s1 = pd.Series(Distance_NS_ALL, name='NS')
    s2 = pd.Series(Distance_S_ALL, name='S')
    df = pd.concat([s1,s2], axis=1)
    # sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], ci=95, capsize=0.05, errwidth=2)
    
    # Freezes
    plt.subplot(2,2,4)
    plt.title('Freezes')
    for i in range(num_cohorts):
        s1 = pd.Series(Freezes_NS_COHORT_list[i], name='NS')
        s2 = pd.Series(Freezes_S_COHORT_list[i], name='S')
        df = pd.concat([s1,s2], axis=1)
        # sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray", color=colors[i])
        sns.violinplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray", color=colors[i])
        
    s1 = pd.Series(Freezes_NS_ALL, name='NS')
    s2 = pd.Series(Freezes_S_ALL, name='S')
    df = pd.concat([s1,s2], axis=1)
    # sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], ci=95, capsize=0.05, errwidth=2)
    plt.suptitle(gene)
    if save:
        filename=filename = analysisFolder + '/' + gene + '_BehaviourSummary.png'
        plt.savefig(filename,dpi=600)
print('FIN')
# FIN