# -*- coding: utf-8 -*-
"""
 SZ_summary:
     - Social Zebrafish - Summary Analysis Functions

@author: adamk
"""
# -----------------------------------------------------------------------------
# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal
import SCZ_analysis as SCZA


#-----------------------------------------------------------------------------
# Utilities for summarizing and plotting "Social Zebrafish" experiments
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
def plotSummaryBoutLocations(ns,s,filename,title,save=True,keep=False,alpha=0.2):
    # All Bouts Summary Plot
    plt.figure()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplot(1,2,1)
    plt.title('NonSocial Phase')
    plt.plot(ns[:, 1], ns[:, 2], '.', color=[0.0, 0.0, 0.0, alpha])
    plt.axis([0, 17, 0, 42])
    plt.gca().invert_yaxis()
        
    plt.subplot(1,2,2)
    plt.title('Social Phase')
    plt.plot(s[:, 1], s[:, 2], '.', color=[0.0, 0.0, 0.0, alpha])
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
# Compute activity level of the fish in bouts per second (BPS)
def measure_BPS(motion, startThreshold, stopThreshold):
                   
    # Find bouts starts and stops
    boutStarts = []
    boutStops = []
    moving = 0
    for i, m in enumerate(motion):
        if(moving == 0):
            if m > startThreshold:
                moving = 1
                boutStarts.append(i)
        else:
            if m < stopThreshold:
                moving = 0
                boutStops.append(i)
    
    # Extract all bouts (ignore last, if clipped)
    boutStarts = np.array(boutStarts)
    boutStops = np.array(boutStops)
    if(len(boutStarts) > len(boutStops)):
        boutStarts = boutStarts[:-1]

    # Count number of bouts
    numBouts= len(boutStarts)
    numberOfSeconds = np.size(motion)/100   ## Assume 100 Frames per Second

    # Set the bouts per second (BPS)
    boutsPerSecond = numBouts/numberOfSeconds
    
    # Measure averge bout trajectory
    boutStarts = boutStarts[(boutStarts > 25) * (boutStarts < (len(motion)-75))]
    allBouts = np.zeros([len(boutStarts), 100])
    for b in range(0,len(boutStarts)):
        allBouts[b,:] = motion[(boutStarts[b]-25):(boutStarts[b]+75)];
    avgBout = np.mean(allBouts,0);

    return boutsPerSecond, avgBout

# Build a histogram of all orientation values
def ort_histogram(ort):

    # ORIENTATION ---------------------------
    numOrts = 36
    interval = 360/numOrts
    ortRange = np.arange(-180,180+interval, interval)    
    ortHistogram, bins = np.histogram(ort, ortRange)

    return ortHistogram

# Analyze bouts and pauses (individual stats)
def analyze_bouts_and_pauses(tracking, testROI, stimROI, visibleFrames, startThreshold, stopThreshold):
    
    # Extract tracking details
    bx = tracking[:,2]
    by = tracking[:,3]
    ort = tracking[:,7]
    motion = tracking[:,8]                
    
    # Compute normlaized arena coordinates
    nx, ny = SCZA.normalized_arena_coords(bx, by, testROI, stimROI)
    
    # Find bouts starts and stops
    boutStarts = []
    boutStops = []
    moving = 0
    for i, m in enumerate(motion):
        if(moving == 0):
            if m > startThreshold:
                moving = 1
                boutStarts.append(i)
        else:
            if m < stopThreshold:
                moving = 0
                boutStops.append(i)
    
    # Extract all bouts (ignore last, if clipped)
    boutStarts = np.array(boutStarts)
    boutStops = np.array(boutStops)
    if(len(boutStarts) > len(boutStops)):
        boutStarts = boutStarts[:-1]
        
    # Check bouts have been found
    if len(boutStarts)==0:
        print('Bad Fish - no bouts detected')
        bouts=[-1]
        pauses=[-1]
    else:

        # Extract all bouts (startindex, startx, starty, startort, stopindex, stopx, stopy, stoport, duration)
        numBouts= len(boutStarts)
        bouts = np.zeros((numBouts, 10))
        for i in range(0, numBouts):
            bouts[i, 0] = boutStarts[i]
            bouts[i, 1] = nx[boutStarts[i]]
            bouts[i, 2] = ny[boutStarts[i]]
            bouts[i, 3] = ort[boutStarts[i]]
            bouts[i, 4] = boutStops[i]
            bouts[i, 5] = nx[boutStops[i]]
            bouts[i, 6] = ny[boutStops[i]]
            bouts[i, 7] = ort[boutStops[i]]
            bouts[i, 8] = boutStops[i] - boutStarts[i]
            bouts[i, 9] = visibleFrames[boutStarts[i]]
            
        # Analyse all pauses (startindex, startx, starty, startort, stopindex, stopx, stopy, stoport, duration)
        numPauses = numBouts+1
        pauses = np.zeros((numPauses, 10))
    
        # -Include first and last as pauses (clipped in video)
        # First Pause
        pauses[0, 0] = 0
        pauses[0, 1] = nx[0]
        pauses[0, 2] = ny[0]
        pauses[0, 3] = ort[0]
        pauses[0, 4] = boutStarts[0]
        pauses[0, 5] = nx[boutStarts[0]]
        pauses[0, 6] = ny[boutStarts[0]]
        pauses[0, 7] = ort[boutStarts[0]]
        pauses[0, 8] = boutStarts[0]
        pauses[0, 9] = visibleFrames[0]
        # Other pauses
        for i in range(1, numBouts):
            pauses[i, 0] = boutStops[i-1]
            pauses[i, 1] = nx[boutStops[i-1]]
            pauses[i, 2] = ny[boutStops[i-1]]
            pauses[i, 3] = ort[boutStops[i-1]]
            pauses[i, 4] = boutStarts[i]
            pauses[i, 5] = nx[boutStarts[i]]
            pauses[i, 6] = ny[boutStarts[i]]
            pauses[i, 7] = ort[boutStarts[i]]
            pauses[i, 8] = boutStarts[i] - boutStops[i-1]
            pauses[i, 9] = visibleFrames[boutStops[i-1]]
        # Last Pause
        pauses[-1, 0] = boutStops[-1]
        pauses[-1, 1] = nx[boutStops[-1]]
        pauses[-1, 2] = ny[boutStops[-1]]
        pauses[-1, 3] = ort[boutStops[-1]]
        pauses[-1, 4] = len(motion)-1
        pauses[-1, 5] = nx[-1]
        pauses[-1, 6] = ny[-1]
        pauses[-1, 7] = ort[-1]
        pauses[-1, 8] = len(motion)-1-boutStops[-1]
        pauses[-1, 9] = visibleFrames[boutStops[-1]]
    return bouts, pauses
        
# Analyze temporal bouts
def analyze_temporal_bouts(bouts, binning):

    # Determine total bout counts
    num_bouts = bouts.shape[0]

    # Determine largest frame number in all bouts recordings (make multiple of 100)
    max_frame = np.int(np.max(bouts[:, 4]))
    max_frame = max_frame + (binning - (max_frame % binning))
    max_frame = 100 * 60 * 15 # 15 minutes

    # Temporal bouts
    visible_bout_hist = np.zeros(max_frame)
    non_visible_bout_hist = np.zeros(max_frame)
    frames_moving = 0
    visible_frames_moving = 0
    non_visible_frames_moving = 0
    for i in range(0, num_bouts):
        # Extract bout params
        start = np.int(bouts[i][0])
        stop = np.int(bouts[i][4])
        duration = np.int(bouts[i][8])
        visible = np.int(bouts[i][9])

        # Ignore bouts beyond 15 minutes
        if stop >= max_frame:
            continue

        # Accumulate bouts in histogram
        if visible == 1:
            visible_bout_hist[start:stop] = visible_bout_hist[start:stop] + 1
            visible_frames_moving += duration
        else:
            non_visible_bout_hist[start:stop] = non_visible_bout_hist[start:stop] + 1
            non_visible_frames_moving += duration
        frames_moving += duration

    #plt.figure()
    #plt.plot(visible_bout_hist, 'b')
    #plt.plot(non_visible_bout_hist, 'r')
    #plt.show()

    # Bin bout histograms
    visible_bout_hist_binned = np.sum(np.reshape(visible_bout_hist.T, (binning, -1), order='F'), 0)
    non_visible_bout_hist_binned = np.sum(np.reshape(non_visible_bout_hist.T, (binning, -1), order='F'), 0)

    #plt.figure()
    #plt.plot(visible_bout_hist_binned, 'b')
    #plt.plot(non_visible_bout_hist_binned, 'r')
    #plt.show()

    # Compute Ratio
    total_bout_hist_binned = visible_bout_hist_binned + non_visible_bout_hist_binned
    vis_vs_non = (visible_bout_hist_binned - non_visible_bout_hist_binned) / total_bout_hist_binned

    # Normalize bout histograms
    #visible_bout_hist_binned = visible_bout_hist_binned / frames_moving
    #non_visible_bout_hist_binned = non_visible_bout_hist_binned / frames_moving
    #vis_v_non = visible_bout_hist_binned / non_visible_bout_hist_binned

    # ----------------
    # Temporal Bouts Summary Plot
    #plt.figure()
    #plt.plot(vis_vs_non, 'k')
    #plt.ylabel('VPI')
    #plt.xlabel('minutes')
    #plt.show()

    return vis_vs_non

# FIN
