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
from scipy.signal import savgol_filter
from scipy.stats import norm

# Breaks a dataframe of motion traces generated from a 96well habituation experiemnt into segments around stimulus frames
def segment_habituation(motion_df, stim_frames, segment_before=40, segment_after=60, frame_rate=100, max_stim=16, offset=10):
    num_fish = motion_df.shape[1]
    num_stimuli = len(stim_frames)
    if num_stimuli>max_stim:
        num_stimuli=max_stim
    parsed_traces = np.zeros((segment_before + segment_after, num_fish, num_stimuli))

    time_index = np.arange(-segment_before, segment_after) / frame_rate  # Time index in seconds

    for fish_num in range(num_fish):
        for stim_num, stim_frame in enumerate(stim_frames):
            actual_stim_frame = stim_frame + offset  # Adjust for actual stimulus frame
            start_frame = actual_stim_frame - segment_before
            end_frame = actual_stim_frame + segment_after
            if end_frame<motion_df.shape[0]:
            
                segment = motion_df.iloc[start_frame:end_frame, fish_num]              
                parsed_traces[:, fish_num, stim_num] = segment

    return time_index, parsed_traces

# Plot all the traces of a given fish
def plot_fish_traces(time_trace,pulse_traces, prepulse_traces, fish_num, colors=['black','magenta'],labels=['Pulse','PrePulse']):
   
   
    if fish_num==0:
        print('ERROR: Fish_num cannot be 0, as this is not a fish, but the time index')
        return -1
    
    # plt.figure(figsize=(10, 6))
    for stim_num in range(pulse_traces.shape[2]):
        pulse_trace = pulse_traces[:, fish_num, stim_num]
        plt.plot(time_trace, pulse_trace,color=colors[0],alpha=0.6)
    
    for stim_num in range(prepulse_traces.shape[2]):
        prepulse_trace = prepulse_traces[:,fish_num,stim_num]
        plt.plot(time_trace, prepulse_trace,color=colors[1],alpha=0.6)
        
    average_trace_pulse = np.nanmean(pulse_trace[:, fish_num, :], axis=1)
    plt.plot(time_trace, average_trace_pulse, label=labels[0], color=colors[0], linewidth=2)
    average_trace_prepulse = np.nanmean(prepulse_trace[:, fish_num, :], axis=1)
    plt.plot(time_trace, average_trace_prepulse, label=labels[1], color=colors[1], linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Motion')
    plt.title(f"Traces for Fish {fish_num}")
    # Show only the "Average" trace in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend()
    
    plt.show()
    plt.figure()
    plt.plot(time_trace, average_trace_pulse, label=labels[0], color=colors[0], linewidth=2)
    plt.plot(time_trace, average_trace_prepulse, label=labels[1], color=colors[1], linewidth=2)
    plt.show()
    
# Compute area under the curve for first 30ms (3 frames) following the stimulus
def compute_integral(parsed_traces, num_frames=3):
    time_trace = parsed_traces[:, 0, :]  # Time trace from parsed_traces
    
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

def compute_smoothed_maximum(time_trace,parsed_traces,filter_length=3, poly_order=2, fullRange=False,frame_range=3, prepulse_delay_frames=30):
    
    pulse_maximums = []
    prepulse_maximums = []
    
    for fish_num in range(parsed_traces.shape[1]):
        pulse_maximum_per_stim = []
        prepulse_maximum_per_stim = []
        
        stim_onset_index = np.where(time_trace == 0)[0][0]  # Find the index where time index is zero
        
        # Compute pulse maximums
        for stim_num in range(parsed_traces.shape[2]):
            subtrace=parsed_traces[:,fish_num,stim_num]
            smoothed_trace = savgol_filter(subtrace, filter_length, poly_order)
            if fullRange:
                trace = smoothed_trace[stim_onset_index :]
            else:
                trace = smoothed_trace[stim_onset_index : stim_onset_index + frame_range]
            maximum = np.nanmax(trace)
            pulse_maximum_per_stim.append(maximum)
        pulse_maximums.append(pulse_maximum_per_stim)
        
        # Compute prepulse maximums
        for stim_num in range(parsed_traces.shape[2]):
            subtrace=parsed_traces[:,fish_num,stim_num]
            smoothed_trace = savgol_filter(subtrace, filter_length, poly_order)
            if fullRange:
                trace = smoothed_trace[ : stim_onset_index]
            else:
                trace = smoothed_trace[stim_onset_index - prepulse_delay_frames : stim_onset_index - prepulse_delay_frames + frame_range]
            maximum = np.nanmax(trace)
            prepulse_maximum_per_stim.append(maximum)
        prepulse_maximums.append(prepulse_maximum_per_stim)
        
    return np.array(pulse_maximums), np.array(prepulse_maximums)

def plot_smoothed_first_16_stimuli(time,parsed_traces, max_values, fish_num, window_length=5, polyorder=2):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot smoothed traces in the first subplot
    ax1.set_title(f"Smoothed First 16 Stimuli Traces for Fish {fish_num}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")

    for stim_num in range(15):
        trace = parsed_traces[:, fish_num, stim_num]
        smoothed_trace = savgol_filter(trace, window_length, polyorder) # Apply smoothing
        smoothed_trace[smoothed_trace<0]=0
        alpha = 0.2 + 0.05 * stim_num  # Adjust transparency
        ax1.plot(time, smoothed_trace, color='black', alpha=alpha)

    avg_trace = np.mean(parsed_traces[:, fish_num, 1:], axis=1)
    avg_smoothed_trace = savgol_filter(avg_trace, window_length, polyorder)  # Apply smoothing
    avg_smoothed_trace[avg_smoothed_trace<0]=0
    ax1.plot(time, avg_smoothed_trace, color='black', linewidth=2, label='Average')

    ax1.legend()

    # Plot max values in the second subplot
    ax2.set_title(f"Max Values for Fish {fish_num}")
    ax2.set_xlabel("Stimulus Number")
    ax2.set_ylabel("Max Value")

    max_values_fish = max_values[fish_num, :16]  # Max values for the first 16 stimuli
    ax2.plot(np.arange(1, 17), max_values_fish, marker='o', color='blue')

    plt.tight_layout()
    plt.show()# Plot a given fish's integral as stimulus number increases
    
def plot_integral_by_stimulus(integrals, fish_num, newFig=True):

    if newFig:
        plt.figure(figsize=(10, 6))
        
    plt.plot(integrals[fish_num, :], marker='o', label=f'Fish {fish_num}')
    
    plt.xlabel('Stim Number')
    plt.ylabel('Integral')
    plt.title("Integral of Response within 20ms following Stimulus")
    # plt.legend()
    plt.show()
    axes=plt.gca()
    return axes

def separate_pulse_prepulse_frames(stim_path,prepulse_interval_ms=300, frame_rate=100, prepulse_range_width_frames=1):
    
    prepulseframerange=int((prepulse_interval_ms/1000)*frame_rate)
    prepulserange=[prepulseframerange-prepulse_range_width_frames,prepulseframerange+prepulse_range_width_frames]
    # Load stim_frames and amplitudes from CSV
    csv = pd.read_csv(stim_path, header=None)
    stim_frames = csv[0]

    prepulse_frames = []
    pulse_frames = []

    for i, stim_frame in enumerate(stim_frames):
        if i > 0:
            prev_stim_frame = stim_frames[i - 1]
            abs_diff_prev = abs(stim_frame - prev_stim_frame)
            if prepulserange[0] <= abs_diff_prev <= prepulserange[1]:
                prepulse_frames.append(stim_frame)

            if i < len(stim_frames) - 1:
                next_stim_frame = stim_frames[i + 1]
                abs_diff_next = abs(next_stim_frame - stim_frame)
                if abs_diff_next < prepulserange[0] or abs_diff_next > prepulserange[1]:
                    if stim_frame not in prepulse_frames:
                        pulse_frames.append(stim_frame)
                        
    return pulse_frames, prepulse_frames

def plot_distribution(data_array, name, hist_range='Auto'):
    flat_data = data_array.flatten()  # Flatten the 2D array into a 1D array
    non_zero_flat_data = flat_data[flat_data != 0]  # Filter out zero values
    
    if hist_range == 'Auto':
        median_overall = np.median(non_zero_flat_data)
        std_overall = np.std(non_zero_flat_data)
        hist_range = [0, median_overall + 5 * std_overall]  # Start from 0
    
    plt.figure(name)
    pdf, bins, _ = plt.hist(flat_data, range=hist_range, bins=20, density=True, edgecolor='black', alpha=0.7)
    bin_width = bins[1] - bins[0]
    pdf_normalized = pdf / (np.sum(pdf) * bin_width)
    plt.plot(bins[:-1], pdf_normalized, color='red', linewidth=2)
    
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Probability Density Function of Response Amplitudes to Non-Zero Trials')
    plt.show()

def calculate_optimal_bin_size(data_arrays, hist_range=[0, 6000]):
    all_data = np.concatenate(data_arrays)
    iqr = np.percentile(all_data, 75) - np.percentile(all_data, 25)
    bin_size = 2 * iqr * len(all_data) ** (-1/3)
       
    min_range, max_range = hist_range
    data_in_range = all_data[(all_data >= min_range) & (all_data <= max_range)]
    
    bin_size = 2 * iqr * len(data_in_range) ** (-1/3)
    
    return bin_size

def plot_distributions(data_arrays, names=None, artificial_bin=True, hist_range='Auto'):
    if hist_range == 'Auto':
        common_bin_size = calculate_optimal_bin_size(data_arrays)
        median_overall = np.median(np.concatenate(data_arrays))
        std_overall = np.std(np.concatenate(data_arrays))
        hist_range = [median_overall - 5 * std_overall, median_overall + 5 * std_overall]
    else:
        common_bin_size = calculate_optimal_bin_size(data_arrays, hist_range)
    
    bins = np.arange(hist_range[0], hist_range[1] + common_bin_size, common_bin_size)
    
    plt.figure()
    for i, data in enumerate(data_arrays):
        pdf, _ = np.histogram(data, bins=bins, density=True, range=hist_range)
        if names is not None:
            name = names[i]
        else:
            name = ''
        
        # Plot the PDF as a line
        plt.plot(bins[:-1], pdf, alpha=0.4, linewidth=2, label=name)
        
        # Artificial bin for values that are equal to 0
        if artificial_bin:
            plt.bar(common_bin_size * -1, 0.01, common_bin_size, color='red', alpha=0.4, label='No Response')

        plt.title(f'Distribution {i+1}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_distribution_per_trial(data_array, bins=20, hist_range='[0,6000]'):
    num_trials = data_array.shape[1]

    for trial_num in range(num_trials):
        plt.figure()
        plt.hist(data_array[:, trial_num], bins=bins, edgecolor='black', alpha=0.7, range=hist_range)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Values - Trial {trial_num + 1}')
        plt.show()

def create_dot_plot(list1, list2, labels):
    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot each pair of values as a dot and connect with lines
    for y1, y2 in zip(list1, list2):
        ax.plot([0, 1], [y1, y2], marker='o', color='blue')  # y-coordinate is set to 1 for each dot

    # Set y-axis limit
    ax.set_ylim(0.5, 1.5)

    # Set x-axis labels
    ax.set_xticks(list1 + list2)
    ax.set_xticklabels([f'{labels[0]}: {x1}\n{labels[1]}: {x2}' for x1, x2 in zip(list1, list2)])

    # Remove y-axis ticks and labels
    ax.yaxis.set_visible(False)

    # Set plot title and show the plot
    plt.title("Dot Plot with Connected Points")
    plt.tight_layout()
    plt.show()

def split_traces_by_genotype(traces, measures, genotype_csv):
    """Split parsed traces into separate arrays for each unique genotype."""
    genotype_map = pd.read_csv(genotype_csv, header=None)
    unique_genotypes = np.unique(genotype_map.values)
    traces_by_genotype = []
    measures_by_genotype = []
    genotypes=[]
    for gene in unique_genotypes:
        if gene=='None':
            continue
        genotypes.append(gene)
        # print(gene)
        
        traces_temp=[]
        measures_temp=[]
        index=0
        for row in range(genotype_map.shape[0]):
            for col in range(genotype_map.shape[1]):
                if gene==genotype_map.iloc[row, col]:  
                    traces_temp.append(traces[:, index, :])
                    measures_temp.append(measures[index])
                index+=1
        # swap axes as we did this slightly differently to previous parsing
        traces_temp=np.array(np.swapaxes(traces_temp, 0, 1))
        traces_by_genotype.append(traces_temp)
        measures_by_genotype.append(measures_temp)
        
    return traces_by_genotype, measures_by_genotype, genotypes

#%% Script
if __name__ == "__main__":
    
    frame_rate=100
    prepulse_delay_frames=30
    ##################################
    plot=False
    trackXY=False
    ##################################
    fullRange=False
    frame_range=3
    ##################################
    # folderListFile=r'D:\dataToTrack\Habituation\FolderLists\Habituation_1000K.txt'
    folderListFile='D:/dataToTrack/PPITrial/FolderLists/PPITrials_230822_PPI.txt'
    data_path,folderNames=PPIU.read_folder_list(folderListFile)
    n=1
    N=len(folderNames)
    
    pulseTracesS,prepulseTracesS=[],[]
    pulseMeasuresS,prepulseMeasuresS, prepulsepreMeasuresS=[],[],[]
    for folder in folderNames:
        print(f'Processing video {n} out of {N}')
        n+=1
        csvs=glob.glob(folder+r'/*.csv')
        parameter_path=[item for item in csvs if "parameters" in item][0]
        left=[item for item in csvs if "parameters" not in item]
        traces_path=[item for item in left if "motionTraces" in item][0]
        left=[item for item in left if "motionTraces" not in item]
        gene_map_path=[item for item in left if "GenotypeMap" in item][0]
        stim_path=[item for item in left if "GenotypeMap" not in item][0]
        # find the stimulus times and amplitudes
        csv=pd.read_csv(stim_path,header=None)
        amplitudes=list(set(csv[2].array))   
        pulse_frames,prepulse_frames=separate_pulse_prepulse_frames(stim_path)
        
        # Grab traces and time index (convert to s)
        df=pd.read_csv(traces_path,header=None)
        traces=np.array(df)
        time_s = [(f / frame_rate) for f in traces[:,0]]
        traces=traces[:,1:]
        
        # Plot the average trace
        if plot:
            plt.figure(figsize=(10, 5))
            average_trace = np.nanmean(traces, axis=1)
            plt.plot(time_s,average_trace, color='b', label='Average Trace')
            for i in range(1,traces.shape[1]): # ignore first stimulus as it does not actually occur (check the bonsai script for this bug)
                plt.plot(time_s,traces[:,i],alpha=0.2)
            for pulse_frame in pulse_frames:
                stim_time=(pulse_frame / frame_rate)
                plt.vlines(stim_time,0,np.nanmax(np.nanmax(traces)),alpha=0.4,color='black')
            for prepulse_frame in prepulse_frames:
                stim_time=(prepulse_frame / frame_rate)
                plt.vlines(stim_time,0,np.nanmean(np.nanmax(traces)),alpha=0.4,color='red')

            plt.xlabel('Time (s)')
            plt.ylabel('Motion AU')
            plt.title('Average Motion Trace of All Fish')
            plt.legend()
            plt.show()
            
        time_seg,pulse_traces=segment_habituation(df, pulse_frames)
        # pulse_response_integrals=compute_integral(pulse_traces,num_frames=3)
        # pulse_maximums=compute_maximum(pulse_traces,num_frames=3)
        pulse_maximums,_=compute_smoothed_maximum(time_seg,pulse_traces,fullRange=fullRange,frame_range=frame_range, prepulse_delay_frames=prepulse_delay_frames)
        pulseTracesS.append(pulse_traces)
        # pulseMeasuresS.append(pulse_response_integrals)
        pulseMeasuresS.append(pulse_maximums)
        
        if len(prepulse_frames)>1:
            _,prepulse_traces=segment_habituation(df, prepulse_frames)
            prepulse_pulse_maximums,prepulse_prepulse_maximums=compute_smoothed_maximum(time_seg,prepulse_traces,fullRange=fullRange,frame_range=frame_range, prepulse_delay_frames=prepulse_delay_frames)
            # prepulse_response_integrals=compute_integral(prepulse_traces,num_frames=3)
            prepulseTracesS.append(prepulse_traces)
            prepulseMeasuresS.append(prepulse_pulse_maximums)
            prepulsepreMeasuresS.append(prepulse_prepulse_maximums)
    
        # if plot and prepulse:
        #     plot_prepulse_vs_pulse(prepulse_traces,pulse_traces)
        # # elif plot:
#%% Debug and analyse
pPAs=[1,0.01]
PAs=[10,0.1]
# pulseMeans, prepulseMeans, prePrepulseMeans = [], [], []
for n, folder in enumerate(folderNames):
    plt.close('all')
    folder=folderNames[n]   
    saveFolder=folder+'/Figures'
    SCZU.cycleMkDir(saveFolder)
    csvs=glob.glob(folder+r'/*.csv')
    parameter_path=[item for item in csvs if "parameters" in item][0]
    left=[item for item in csvs if "parameters" not in item]
    traces_path=[item for item in left if "motionTraces" in item][0]
    left=[item for item in left if "motionTraces" not in item]
    stim_path=[item for item in csvs if "parameters" not in item][0]
    # find the stimulus times and amplitudes
    csv=pd.read_csv(stim_path,header=None)
    amplitudes=list(set(csv[2].array))   
    pulse_frames,prepulse_frames=separate_pulse_prepulse_frames(stim_path)
    
    # Grab traces
    df=pd.read_csv(traces_path,header=None)
    traces=np.array(df)
    
    # Ignore first (phantom) stimulus? - Fixed in Bonsai 22/08
    # traces=traces[:,1:]
    pulse_traces=pulseTracesS[n][:,1:,:]
    prepulse_traces=prepulseTracesS[n][:,1:,:]
    
    pulse_measures=pulseMeasuresS[n][1:,:]
    prepulse_measures=prepulseMeasuresS[n][1:,:]
    prepulse_premeasures=prepulsepreMeasuresS[n][1:,:]
    
    ######################### SPLIT INTO GENOTYPES ###########################
    pulse_gene_traces,pulse_gene_measures,genotypes = split_traces_by_genotype(pulse_traces,pulse_measures,gene_map_path)
    prepulse_gene_traces,prepulse_gene_measures,_ = split_traces_by_genotype(prepulse_traces,prepulse_measures,gene_map_path)
    
    for nG, genotype in enumerate(genotypes):
        # grab traces for this genotype
        pulse_traces=np.array(pulse_gene_traces[nG])
        prepulse_traces=np.array(prepulse_gene_traces[nG])
        pulse_measures=np.array(pulse_gene_measures[nG])
        prepulse_measures=np.array(prepulse_gene_measures[nG])
        
        for i in range(pulse_traces.shape[1]):
            for j in range(pulse_traces.shape[2]):
                # 'pulse'
                trace=pulse_traces[:,i,j]
                inter=savgol_filter(trace, window_length=7, polyorder=2)
                inter[inter<0]=0
                pulse_traces[:,i,j]=inter
            
        for i in range(prepulse_traces.shape[1]):
            for j in range(prepulse_traces.shape[2]):
                # 'pulse'
                trace=prepulse_traces[:,i,j]
                inter=savgol_filter(trace, window_length=7, polyorder=2)
                inter[inter<0]=0
                prepulse_traces[:,i,j]=inter
    
    
        # Ensure we have the same number of stimuli for each trial type and trim longer one if needed (throws concatenation error later if not)
        if pulse_traces.shape[2]<prepulse_traces.shape[2]:
            nn=prepulse_traces.shape[2]-pulse_traces.shape[2]
            prepulse_traces=prepulse_traces[:,:,:-nn]
            prepulse_measures=prepulse_measures[:,:-nn]
            prepulse_premeasures=prepulse_premeasures[:,:-nn]
        elif pulse_traces.shape[2]>prepulse_traces.shape[2]:
            nn=pulse_traces.shape[2]-prepulse_traces.shape[2]
            pulse_traces=pulse_traces[:,:,:-nn]
            pulse_measures=pulse_measures[:,:-nn]
        
        plot_distribution(pulse_measures,'Pulse')
        # plt.ylim(0,550)
        plt.savefig(saveFolder+'/Pulse_amplitudes_P' + str(PAs[n]) + '_pP' + str(pPAs[n]) + '_' + str(genotype) + '%.png', dpi=600)
        
        plot_distribution(prepulse_measures,'PrePulse-Pulse')
        # plt.ylim(0,550)
        plt.savefig(saveFolder+'/PrePulse_Pulse_amplitudes_P' + str(PAs[n]) + '_pP' + str(pPAs[n]) + '_' + str(genotype) + '%.png', dpi=600)
        
        plot_distribution(prepulse_premeasures,'PrePulse-PrePulse')
        # plt.ylim(0,550)
        plt.savefig(saveFolder+'/PrePulse_Amplitudes_P' + str(PAs[n]) + '_pP' + str(pPAs[n]) + '_' + str(genotype) + '%.png', dpi=600)
        plt.close('all')
        
        # data=[pulse_measures,prepulse_premeasures,prepulse_measures]
        # plot_distributions(data,['Pulse','Prepulse_PrePulse','Prepulse_Pulse'],0)
        # plt.savefig(saveFolder+'/AllPDFS_' + str(PAs[n]) + '_pP' + str(pPAs[n]) + '%.png', dpi=600)
        # plot_distribution(data,['Pulse','Prepulse_PrePulse','Prepulse_Pulse'],0)
        
        plt.figure()
        pulseMeanPerTrial=np.mean(pulse_traces,axis=1)
        pulseMeanPerPlate=np.mean(pulseMeanPerTrial,axis=1)
        prepulseMeanPerTrial=np.mean(prepulse_traces,axis=1)
        prepulseMeanPerPlate=np.mean(prepulseMeanPerTrial,axis=1)
        PPI_ReductionRatio_Plate=1-np.max(prepulseMeanPerPlate[40:])/np.max(pulseMeanPerPlate[40:])
        
        print('PPI Ratio for gene ' + str(genotype) + ' P' + str(PAs[n]) + ' pP' + str(pPAs[n]) + f' = {PPI_ReductionRatio_Plate}')
        plt.plot(time_seg,pulseMeanPerPlate,color='black', label='Pulse')
        plt.plot(time_seg,prepulseMeanPerPlate,color='magenta', label='PrePulse')
        plt.legend()
        plt.xlabel('Time relative to pulse (s)')
        plt.ylabel('Motion (AU)')
        plt.savefig(saveFolder+'/Plate_motion_per_trial_P' + str(PAs[n]) + '_pP' + str(pPAs[n]) + '_' + str(genotype) + '%.png', dpi=600)
        
        plt.figure()
        plt.plot(time_seg,pulseMeanPerTrial,color='black')
        plt.plot(time_seg,prepulseMeanPerTrial,color='magenta')
        plt.xlabel('Time relative to pulse (s)')
        plt.ylabel('Motion (AU)')
        plt.savefig(saveFolder+'/Plate_MeanMotion_P' + str(PAs[n]) + '_pP' + str(pPAs[n]) + '_' + str(genotype) + '%.png', dpi=600)
        
    # plt.ylim(0,2500)
    # def plot_prepulse_vs_pulse(prepulse_traces,pulse_traces):
    #     # Calculate the average traces for prepulse and pulse trials
    #     avg_prepulse_trace = np.mean(prepulse_traces, axis=0)
    #     avg_pulse_trace = np.mean(pulse_traces, axis=0)
    
    #     # Set up the figure and axis
    #     plt.figure(figsize=(10, 6))
    #     plt.title("Aligned Traces for Prepulse and Pulse Trials")
    #     plt.xlabel("Time")
    #     plt.ylabel("Amplitude")
    #     time=pulse_traces[0]
    #     # Plot individual traces with transparency
    #     for trace in prepulse_traces[1:]:
    #         plt.plot(time,trace, color='blue', alpha=0.4)
        
    #     for trace in pulse_traces[1:]:
    #         plt.plot(time,trace, color='orange', alpha=0.4)
        
    #     # Plot average traces with thicker lines
    #     plt.plot(time,avg_prepulse_trace, color='blue', linewidth=2, label='Avg Prepulse')
    #     plt.plot(time,avg_pulse_trace, color='orange', linewidth=2, label='Avg Pulse')
        
    #     # Add legend
    #     plt.legend()
        
    #     # Show the plot
    #     plt.show()
