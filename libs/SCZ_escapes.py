# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:50:24 2022

@author: Tom
"""
lib_path =r'S:\\WIBR_Dreosti_Lab\\Tom\\Github\\Arena_Zebrafish\\libs'
ARK_lib_path =r'S:\\WIBR_Dreosti_Lab\\Tom\\Github\\Arena_Zebrafish\\ARK\\libs'
import sys
sys.path.append(lib_path)
sys.path.append(ARK_lib_path)
import numpy as np
import ARK_utilities as ARKU
import AZ_utilities as AZU
import AZ_compare as AZC
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd

# load escape traces
# esc_file='D:/DataForAdam/TrackingData/EC_B2/Analysis_NEWER/Looms/210218_EmxGFP_Ctrl_B2_0_escape_traces.npy'


#%% compute bout signals def
def compute_smooth_bout_signal(speed_space,speed_angle,motion):

    # Absolute Value of angular speed
    speed_abs_angle = np.abs(speed_angle)
    
    # Detect negative/error values and set to zero
    bad_values = (motion < 0) + (speed_space < 0)
    speed_space[bad_values] = 0.0
    speed_abs_angle[bad_values] = 0.0
    motion[bad_values] = 0.0
    
    # Weight contribution by STD
    std_space = np.std(speed_space)    
    std_angle = np.std(speed_abs_angle)    
    speed_space_norm = speed_space/std_space
    speed_angle_norm = speed_abs_angle/std_angle
    # std_motion = np.std(motion)
    # motion_norm = motion/std_motion
    
    # Sum weighted signals
    bout_signal = speed_space_norm + speed_angle_norm# + motion_norm
    
    # Interpolate over bad values
    for i, bad_value in enumerate(bad_values):
        if bad_value == True:
            bout_signal[i] = bout_signal[i-1]
    
    # Smooth signal for bout detection   
    bout_filter = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    smooth_bout_signal = signal.fftconvolve(bout_signal, bout_filter, 'same')  
    
    # Determine Threshold levels
    # - Determine the largest 15 values and take the median
    # - Use 10% of max level, divide by 10, for the base threshold
    sorted_bout_signal = np.sort(smooth_bout_signal)
    max_norm = np.median(sorted_bout_signal[-15:])    
    upper_threshold = max_norm/10
    lower_threshold = upper_threshold/2 
    return smooth_bout_signal, upper_threshold, lower_threshold

#%% 
def parseEscapes(esc_trace_file,numBoutsToCount=2):
    
    esc_traces=np.load(esc_trace_file)    
    ballistic_boutsS,next_boutsS=[],[]
    
    # loop through escape traces for this fish
    numEscapes=len(esc_traces)
    # if esc_traces!=-1:
    
    for k in range(numEscapes):
        # find motion signal as we do for bouts
        speed_space=esc_traces[k,:,5]
        motion=esc_traces[k,:,6]
        speed_angle=esc_traces[k,:,7]
        speed_angle=speed_angle
        
        smooth_bout_signal, upper_threshold, lower_threshold = compute_smooth_bout_signal(speed_space,speed_angle,motion)
        starts, peaks, stops = ARKU.find_escape_peaks_dual_threshold(smooth_bout_signal, upper_threshold, lower_threshold)
        numBouts=len(peaks)
        # Record metrics of first "ballistic" escape bout
        next_bouts = np.zeros([numBoutsToCount, 9])
        # load rotTrajX and rotTrajY coordinates 
        bx=esc_traces[k,:,2]
        by=esc_traces[k,:,3]
            
        # Set bout parameters
        ballistic_bouts = []
        search_ballistic=True
        boutCount=0
        
        
        for i in range(numBouts): 
            if i < len(starts):
                start = starts[i]   # Start frame
                stop = stops[i]         # Stop frame
            
                x = bx[start:stop]      # X trajectory
                y = by[start:stop]      # Y trajectory
            
                if search_ballistic and smooth_bout_signal[peaks[i]]>4:
                    # print('found ballistic bout')
                    xLoom=x[0]
                    yLoom=y[0]
                    ballistic_bouts.append(starts[i]) # delay to ballistic response
                    ballistic_bouts.append(peaks[i])      # Peak frame
                    ballistic_bouts.append(stops[i])    # frame of Lower threshold crossing
                    ballistic_bouts.append(stops[i]-starts[i]) # Durations
                    ballistic_bouts.append(np.sum(speed_angle[starts[i]:stops[i]])/120) # Net angle change  
                    ballistic_bouts.append(np.sqrt(x[-1]**2 + y[-1]**2)*0.09) # Net distance change
                    ballistic_bouts.append(x[-1]) # final relative x location of this bout
                    ballistic_bouts.append(y[-1]) # final relative y location of this bout
                    search_ballistic=False
                    
                elif search_ballistic==False and boutCount<numBoutsToCount-1:
                    # print('found second bout')
                    next_bouts[boutCount, 0] = starts[i] # start of next bout relative to start of loom
                    next_bouts[boutCount, 1] = peaks[i]      # Peak frame
                    next_bouts[boutCount, 2] = stops[i]+1    # frame of Lower threshold crossing
                    next_bouts[boutCount, 3] = stops[i]-starts[i] # Durations
                    next_bouts[boutCount, 4] = np.sum(speed_angle[starts[i]:stops[i]]/120) # Net angle change  
                    next_bouts[boutCount, 5] = np.sqrt(x[-1]**2 + y[-1]**2)*0.09 # Net distance change
                    next_bouts[boutCount, 6] = x[-1] # final x position of the trajectory of this bout reltive to start
                    next_bouts[boutCount, 7] = y[-1] # final y position of the trajectory of this bout reltive to start
                    next_bouts[boutCount, 8] = np.sqrt(((xLoom-x[-1])**2) + ((yLoom-y[-1])**2))*0.09 # Net distance change from Loom
                    
                    boutCount+=1
                                
        # collect all metrics for this trace
        ballistic_boutsS.append(np.array(ballistic_bouts))
        next_boutsS.append(next_bouts)
        
    return ballistic_boutsS,next_boutsS

# write function to count the number of ballistic escapes that are folowed by another up to two bouts within 2 seconds of loom stimulus being absent
def proportionOfExtraBouts(ballistic_bouts,next_bouts):
    count_next=0
    count_second=0
    # for each ballistic bout,check if there was a follow up bout
    for i in range(len(ballistic_bouts)):
        if next_bouts[i][0][0]>0: # if start is zero, it's not a bout
            count_next+=1
            if next_bouts[i][1][0]>0: # if there was a followup bout, check if there's a second
                count_second+=1
                    
    # return proportion of ballistic escapes that have one, two or three follow up bouts
    propNext=np.divide(count_next,len(ballistic_bouts))
    propSecond=np.divide(count_second,len(ballistic_bouts))

    return propNext,propSecond

#%% Script

import glob
escapeFolder_C='D:\Data\Artur_Data\Analysis_Artur\Mn_WT\TrackingData\Analysis\Looms'
escapeFolder_A='D:\Data\Artur_Data\Analysis_Artur\Mn_mutant\TrackingData\Analysis\Looms'
escapeFolders=[escapeFolder_C,escapeFolder_A]

ballistic_boutsSS,next_boutsSS,second_boutsSS,propNextSS,propSecondSS=[],[],[],[],[]

# LOOP THROUGH GROUPS
for escapeFolder in escapeFolders:
    esc_files=glob.glob(escapeFolder+'\\*escapes*')
    esc_trace_files=glob.glob(escapeFolder+'\\*escape_trace*')
    numFish=len(esc_files)
    
    ballistic_boutsS,next_boutsS,second_boutsS,propNextS,propSecondS=[],[],[],[],[]
    # LOOP THROUGH FISH
    for j in range(numFish):
        esc_trace_file=esc_trace_files[j]
        esc_file=esc_files[j]
        ballistic_bouts,next_bouts=parseEscapes(esc_trace_file)
        propNext,propSecond=proportionOfExtraBouts(ballistic_bouts,next_bouts)
        ballistic_boutsS.append(ballistic_bouts)
        propNextS.append(propNext)
        propSecondS.append(propSecond)
        
        next_bouts_list,second_bouts_list=[],[]
        # LOOP THROUGH NEXT BOUTS (maybe only those that occur after 1 sec (when stimulus has disappeared))
        for k in range(len(next_bouts)):
            # for l,bout in enumerate(next_bouts):
            #     if bout[0,0]>120:
            l=0
            next_bouts_list.append(next_bouts[k][l,:])
            second_bouts_list.append(next_bouts[k][l+1,:])
        # Collect fish
        next_boutsS.append(np.array(next_bouts_list))
        second_boutsS.append(np.array(second_bouts_list))
    # Collect groups
    ballistic_boutsSS.append(ballistic_boutsS)
    next_boutsSS.append(next_boutsS)
    second_boutsSS.append(second_boutsS)
    propNextSS.append(propNextS)
    propSecondSS.append(propSecondS)
# Now we have everything nice and loaded, let's plot some things

saveFolder=r'D:\\Data\Artur_Data\\Analysis_Artur\\LoomFigures'
AZU.cycleMkDir(saveFolder)
save=True
SE=True
dist_yint=[0,10,20,30,40,50]
angle_yint=[0,20,40,60,80,100,120,140,160,180,200,220,240,260]
################## Average distance of ballistic escapes ###############
escapes_con=ballistic_boutsSS[0]
escapes_les=ballistic_boutsSS[1]

avDistsSS=[]
avDistsS=[]
# loop through fish in control group
for fish in escapes_con:
    dists=[]
    for escape in fish:
        if escape.size!=0:
            dist=escape[5]
            if dist<10000:
                dists.append(dist)
    avDistsS.append(np.mean(dists))
avDistsSS.append(avDistsS)
# loop through fish in lesion group        
avDistsS=[]
for fish in escapes_les:
    dists=[]
    for escape in fish:
        if escape.size!=0:
            dist=escape[5]
            if dist<10000:
                dists.append(dist)
    avDistsS.append(np.mean(dists))
avDistsSS.append(avDistsS)

groups=avDistsSS
labels=['Control','Lesion']
figname='Ballistic - Average net displacement'
savepath=saveFolder+'/ballistic_displacement.png'
yint=dist_yint
ylabel='Total distance (mm)'
ylim=[0,np.max(yint)]
AZC.compPlot(groups[0],groups[1],labels,figname,savepath,yint,ylabel,ylim,save=save,SE=SE)  

################## Average displacement of next escapes ###############
escapes_con=next_boutsSS[0]
escapes_les=next_boutsSS[1]

avDistsSS=[]
avDistsS=[]
# loop through fish in control group
for fish in escapes_con:
    dists=[]
    for escape in fish:
        if escape.size!=0 and escape[5]!=0:
            dist=escape[5]
            if dist<10000:
                dists.append(dist)
    if len(dists)!=0:
        avDistsS.append(np.mean(dists))
avDistsSS.append(avDistsS)
# loop through fish in lesion group        
avDistsS=[]
for fish in escapes_les:
    dists=[]
    for escape in fish:
        if escape.size!=0 and escape[5]!=0:
            dist=escape[5]
            if dist<10000:
                dists.append(dist)
    if len(dists)!=0:
        avDistsS.append(np.mean(dists))
avDistsSS.append(avDistsS)

groups=avDistsSS
labels=['Control','Lesion']
figname='Next bout - Average net displacement'
savepath=saveFolder+'/next_displacement.png'
yint=dist_yint
ylabel='Total distance (mm)'
ylim=[0,np.max(yint)]
AZC.compPlot(groups[0],groups[1],labels,figname,savepath,yint,ylabel,ylim,save=save,SE=SE)  

################## Average displacement of next escapes from loom centre ###############
escapes_con=next_boutsSS[0]
escapes_les=next_boutsSS[1]

avDistsSS=[]
avDistsS=[]
# loop through fish in control group
for fish in escapes_con:
    dists=[]
    for escape in fish:
        if escape.size!=0 and escape[5]!=0:
            dist=escape[8]
            if dist<10000:
                dists.append(dist)
    if len(dists)!=0:
        avDistsS.append(np.mean(dists))
avDistsSS.append(avDistsS)
# loop through fish in lesion group        
avDistsS=[]
for fish in escapes_les:
    dists=[]
    for escape in fish:
        if escape.size!=0 and escape[5]!=0:
            dist=escape[8]
            if dist<10000:
                dists.append(dist)
    if len(dists)!=0:
        avDistsS.append(np.mean(dists))
avDistsSS.append(avDistsS)

groups=avDistsSS
labels=['Control','Lesion']
figname='Next bout - Average net displacement from loom Stim'
savepath=saveFolder+'/next_displacement_from_loom.png'
yint=dist_yint
ylabel='Total distance (mm)'
ylim=[0,np.max(yint)]
AZC.compPlot(groups[0],groups[1],labels,figname,savepath,yint,ylabel,ylim,save=save,SE=SE)  

################## Average distance of second escapes ###############
escapes_con=second_boutsSS[0]
escapes_les=second_boutsSS[1]

avDistsSS=[]
avDistsS=[]
# loop through fish in control group
for fish in escapes_con:
    dists=[]
    for escape in fish:
        if escape.size!=0:
            dist=escape[5]
            if dist<10000:
                dists.append(dist)
    avDistsS.append(np.mean(dists))
avDistsSS.append(avDistsS)
# loop through fish in lesion group        
avDistsS=[]
for fish in escapes_les:
    dists=[]
    for escape in fish:
        if escape.size!=0:
            dist=escape[5]
            if dist<10000:
                dists.append(dist)
    avDistsS.append(np.mean(dists))
avDistsSS.append(avDistsS)

groups=avDistsSS
labels=['Control','Lesion']
figname='Second bout - Average net displacement'
savepath=saveFolder+'/second_displacement.png'
yint=dist_yint
ylabel='Total distance (mm)'
ylim=[0,np.max(yint)]
AZC.compPlot(groups[0],groups[1],labels,figname,savepath,yint,ylabel,ylim,save=save,SE=SE)  

################## Average angle of ballistic escapes ###############
escapes_con=ballistic_boutsSS[0]
escapes_les=ballistic_boutsSS[1]

avDistsSS=[]
avDistsS=[]
# loop through fish in control group
for fish in escapes_con:
    dists=[]
    for escape in fish:
        if escape.size!=0:
            dist=escape[4]
            dists.append(np.abs(dist))
    avDistsS.append(np.mean(dists))
avDistsSS.append(avDistsS)
# loop through fish in lesion group        
avDistsS=[]
for fish in escapes_les:
    dists=[]
    for escape in fish:
        if escape.size!=0:
            dist=escape[4]
            dists.append(np.abs(dist))
    avDistsS.append(np.mean(dists))
avDistsSS.append(avDistsS)

groups=avDistsSS
labels=['Control','Lesion']
figname='Ballistic bout - Average net angle change'
savepath=saveFolder+'/ballistic_angle.png'
yint=angle_yint
ylabel='Total angle (deg)'
ylim=[0,np.max(yint)]
AZC.compPlot(groups[0],groups[1],labels,figname,savepath,yint,ylabel,ylim,save=save,SE=SE)  


################## Average angle of next escapes ###############
escapes_con=next_boutsSS[0]
escapes_les=next_boutsSS[1]

avDistsSS=[]
avDistsS=[]
# loop through fish in control group
for fish in escapes_con:
    dists=[]
    for escape in fish:
        if escape.size!=0:
            dist=escape[4]
            dists.append(np.abs(dist))
    avDistsS.append(np.mean(dists))
avDistsSS.append(avDistsS)
# loop through fish in lesion group        
avDistsS=[]
for fish in escapes_les:
    dists=[]
    for escape in fish:
        if escape.size!=0:
            dist=escape[4]
            dists.append(np.abs(dist))
    avDistsS.append(np.mean(dists))
avDistsSS.append(avDistsS)

groups=avDistsSS
labels=['Control','Lesion']
figname='Next bout - Average net angle change'
savepath=saveFolder+'/next_angle.png'
yint=angle_yint
ylabel='Total angle (deg)'
ylim=[0,np.max(yint)]
AZC.compPlot(groups[0],groups[1],labels,figname,savepath,yint,ylabel,ylim,save=save,SE=SE)  

################## Average angle of second bouts after escape ###############
escapes_con=second_boutsSS[0]
escapes_les=second_boutsSS[1]

avDistsSS=[]
avDistsS=[]
# loop through fish in control group
for fish in escapes_con:
    dists=[]
    for escape in fish:
        if escape.size!=0:
            dist=escape[4]
            dists.append(np.abs(dist))
    avDistsS.append(np.mean(dists))
avDistsSS.append(avDistsS)
# loop through fish in lesion group        
avDistsS=[]
for fish in escapes_les:
    dists=[]
    for escape in fish:
        if escape.size!=0:
            dist=escape[4]
            dists.append(np.abs(dist))
    avDistsS.append(np.mean(dists))
avDistsSS.append(avDistsS)

groups=avDistsSS
labels=['Control','Lesion']
figname='Second bout - Average net angle change'
savepath=saveFolder+'/second_angle.png'
yint=angle_yint
ylabel='Total angle (deg)'
ylim=[0,np.max(yint)]
AZC.compPlot(groups[0],groups[1],labels,figname,savepath,yint,ylabel,ylim,save=save,SE=SE)  


################### Ballistic displacement vs angle scatter
escapes_con=ballistic_boutsSS[0]
escapes_les=ballistic_boutsSS[1]
figname='Ballistic bout displacement vs angle'
distsSS=[]

anglesSS=[]

# loop through fish in control group
dists=[]
angles=[]
for fish in escapes_con:
    for escape in fish:
        if escape.size!=0:
            dist=escape[5]
            angle=escape[4]
            if dist<10000 and dist>0:
                dists.append(dist)
                angles.append(angle)
distsSS.append(dists)
anglesSS.append(angles)
# loop through fish in lesion group      
dists=[]  
angles=[]
for fish in escapes_les:
    for escape in fish:
        if escape.size!=0:
            dist=escape[5]
            angle=escape[4]
            if dist<10000 and dist>0:
                dists.append(dist)
                angles.append(angle)
distsSS.append(dists)
anglesSS.append(angles)

plt.figure(figname)
for i in range(len(distsSS)):
    plt.scatter(anglesSS[i],distsSS[i],alpha=0.3)
plt.title(figname)
plt.ylabel('Displacement (mm)')
plt.xlabel('Angle (o)')
#################### Next displacement vs angle scatter
escapes_con=next_boutsSS[0]
escapes_les=next_boutsSS[1]

figname='Next bout displacement vs angle'
distsSS=[]

anglesSS=[]

# loop through fish in control group
dists=[]
angles=[]
for fish in escapes_con:
    for escape in fish:
        if escape.size!=0:
            dist=escape[5]
            angle=escape[4]
            if dist<10000 and dist>0:
                dists.append(dist)
                angles.append((angle))
distsSS.append(dists)
anglesSS.append(angles)
# loop through fish in lesion group      
dists=[]  
angles=[]
for fish in escapes_les:
    for escape in fish:
        if escape.size!=0:
            dist=escape[5]
            angle=escape[4]
            if dist<10000 and dist>0:
                dists.append(dist)
                angles.append((angle))
distsSS.append(dists)
anglesSS.append(angles)

plt.figure(figname)
for i in range(len(distsSS)):
    plt.scatter(anglesSS[i],distsSS[i],alpha=0.3)
plt.title(figname)
plt.ylabel('Displacement (mm)')
plt.xlabel('Angle (o)')
# def peakToPeakDelay(ballistic_bouts,next_bouts):
    
## Write function for:
# plot the displacements AZC of the ballistic bouts, control vs lesion
# plot the angles AZC of ballistic bouts, control vs lesion
# plot scatter of the ballistic displacements vs angles, control vs lesion

# do the same for the next and second bout following the ballistic response (use function)

# count proportion of escapes that have an additional bout within 3 seconds

# np.sum(next_boutsS)
        # debug
        # plt.figure()
        # plt.plot(smooth_bout_signal)
        # plt.hlines(upper_threshold,0,360,colors='r')
        # plt.hlines(lower_threshold,0,360,colors='blue')
        
        # plt.vlines(starts,0,np.max(smooth_bout_signal),colors='r')
        # plt.vlines(peaks,0,np.max(smooth_bout_signal),colors='black')
        # plt.vlines(stops,0,np.max(smooth_bout_signal),colors='blue')