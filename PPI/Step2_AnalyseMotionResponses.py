# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 13:39:15 2023

@author: Tom
"""
lib_path = r'D:\Tom\Github\SCZ_Fish\libs'
import sys
sys.path.append(lib_path)

import cv2
import glob
import pandas as pd
import numpy as np
import SCZ_utilities as SCZU
import SCZ_video as SCZV
import BONSAI_ARK
import matplotlib.pyplot as plt
import PPI_utilities as PPIU

# Breaks a dataframe of motion traces generated from a 96well habituation experiemnt into segments around stimulus frames
def segment_habituation(motion_df, stim_frames, segment_before=40, segment_after=60, frame_rate=100):
    num_fish = motion_df.shape[1]
    num_stimuli = len(stim_frames)
    
    parsed_traces = np.zeros((segment_before + segment_after, num_fish, num_stimuli))

    time_index = np.arange(-segment_before, segment_after) / frame_rate * 1000  # Time index in milliseconds

    for fish_num in range(num_fish):
        for stim_num, stim_frame in enumerate(stim_frames):
            actual_stim_frame = stim_frame + 12  # Adjust for actual stimulus frame
            start_frame = actual_stim_frame - segment_before
            end_frame = actual_stim_frame + segment_after

            segment = motion_df.iloc[start_frame:end_frame, fish_num]

            parsed_traces[:, fish_num, stim_num] = segment

    # Replace the time index with converted units (0 is stimulus onset)
    for stim_num in range(num_stimuli):
        parsed_traces[:, 0, stim_num] = time_index

    return parsed_traces

# Plot all the traces of a given fish
def plot_fish_traces(parsed_traces, fish_num):
    time_trace = parsed_traces[:, 0, 0]  # Time trace
   
    if fish_num==0:
        print('ERROR: Fish_num cannot be 0, as this is not a fish, but the time index')
        return -1
    
    plt.figure(figsize=(10, 6))
    for stim_num in range(parsed_traces.shape[2]):
        trace = parsed_traces[:, fish_num, stim_num]
        plt.plot(time_trace, trace, label=f"Stimulus {stim_num + 1}")

    average_trace = np.mean(parsed_traces[:, fish_num, :], axis=1)
    plt.plot(time_trace, average_trace, label="Average", color='black', linewidth=2)

    plt.xlabel('Time (ms)')
    plt.ylabel('Motion')
    plt.title(f"Traces for Fish {fish_num}")
    # Show only the "Average" trace in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[-1:], labels[-1:])
    
    plt.show()

# Compute area under the curve for first 30ms (3 frames) following the stimulus
def compute_integral(parsed_traces, num_frames=3):
    time_trace = parsed_traces[:, 0, 0]  # Time trace from parsed_traces
    
    integrals = []
    for fish_num in range(parsed_traces.shape[1]):
        integral_per_stim = []
        stim_onset_index = np.where(time_trace == 0)[0][0]  # Find the index where time index is zero
        for stim_num in range(parsed_traces.shape[2]):
            trace = parsed_traces[stim_onset_index : stim_onset_index + num_frames, fish_num, stim_num]
            integral = np.sum(trace)
            integral_per_stim.append(integral)
        integrals.append(integral_per_stim)
    return np.array(integrals)

# Plot a given fish's integral as stimulus number increases
def plot_integral_by_stimulus(integrals, fish_num, newFig=True):

    if newFig:
        plt.figure(figsize=(10, 6))
        
    plt.plot(integrals[fish_num, :], marker='o', label=f'Fish {fish_num}')
    
    plt.xlabel('Stim Number')
    plt.ylabel('Integral')
    plt.title(f"Integral of Response within 30ms following Stimulus")
    # plt.legend()
    plt.show()
    axes=plt.gca()
    return axes


#%% Script
if __name__ == "__main__":
    
    frame_rate=100
    plot=False
    trackXY=False
    folderListFile=r'D:\dataToTrack\Habituation\FolderLists\Habituation_1000K.txt'
    data_path,folderNames=PPIU.read_folder_list(folderListFile)
    n=1
    N=len(folderNames)
    
    parsedTracesS=[]
    measuresS=[]
    for folder in folderNames:
        print(f'Processing video {n} out of {N}')
        n+=1
        csvs=glob.glob(folder+r'/*.csv')
        parameter_path=[item for item in csvs if "parameters" in item][0]
        left=[item for item in csvs if "parameters" not in item]
        traces_path=[item for item in left if "motionTraces" in item][0]
        left=[item for item in left if "motionTraces" not in item]
        stim_path=[item for item in csvs if "parameters" not in item][0]
        # find the stimulus times and amplitudes
        csv=pd.read_csv(stim_path,header=None)
        stim_frames=csv[0]
        amplitudes=list(set(csv[2].array))   
        
        # Grab traces and time index (convert to ms)
        df=pd.read_csv(traces_path,header=None)
        traces=np.array(df)
        time_s = [(f / frame_rate) for f in traces[:,0]]
        traces=traces[:,1:]
        
        # Plot the average trace
        if plot:
            plt.figure(figsize=(10, 5))
            average_trace = np.mean(traces, axis=1)
            plt.plot(time_s,average_trace, color='b', label='Average Trace')
            for i in range(traces.shape[1]):
                plt.plot(time_s,traces[:,i],alpha=0.2)
            for stim_frame in stim_frames:
                stim_time=(stim_frame / frame_rate)
                plt.vlines(stim_time,0,np.max(np.max(traces)),alpha=0.4,color='black')

            plt.xlabel('Time (s)')
            plt.ylabel('Motion AU')
            plt.title('Average Motion Trace of All Fish')
            plt.legend()
            plt.show()
    
        if len(amplitudes)>1:
            print('ERROR! Appears to be a prepulse experiment!')
        else:
            parsed_traces=segment_habituation(df, stim_frames)
            response_integrals=compute_integral(parsed_traces,num_frames=2) # num_frames is the number of frames after the stimulus onset to take an integral. Default is 2 = 20 ms i.e. enough for the SL responses to be in full swing, but not the LLC onset, which occurs typically after 15-18ms
        
        parsedTracesS.append(parsed_traces)
        measuresS.append(response_integrals)