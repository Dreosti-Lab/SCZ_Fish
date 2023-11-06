# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 13:39:15 2023

@author: Tom
"""

#%% Import and functions
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
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore, ttest_ind, f_oneway, kruskal, norm

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

def smooth_traces(traces):
    num_time_points, num_fish, num_trials = traces.shape

    # Create an empty array to store the smoothed traces
    smoothed_traces = np.empty_like(traces)

    # Loop through each trial and fish to apply the Savitzky-Golay filter
    for trial in range(num_trials):
        for fish in range(num_fish):
            trace = traces[:, fish, trial]
            smoothed_trace = savgol_filter(trace, window_length=5, polyorder=2)
            smoothed_traces[:, fish, trial] = smoothed_trace
    
    smoothed_traces[smoothed_traces < 0] = 0
    return smoothed_traces

def compute_smoothed_maximum(time_trace,parsed_traces,filter_length=5, poly_order=2, fullRange=False,frame_range=3, prepulse_delay_frames=30):
    
    pulse_maximums = []
    prepulse_maximums = []
    smooth_parsed_traces = np.empty_like(parsed_traces)
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
            smoothed_trace[smoothed_trace<0]=0
            smooth_parsed_traces[:,fish_num,stim_num]=smoothed_trace
            if fullRange:
                trace = smoothed_trace[ : stim_onset_index]
            else:
                trace = smoothed_trace[stim_onset_index - prepulse_delay_frames : stim_onset_index - prepulse_delay_frames + frame_range]
            maximum = np.nanmax(trace)
            prepulse_maximum_per_stim.append(maximum)
        prepulse_maximums.append(prepulse_maximum_per_stim)
        
    return np.array(pulse_maximums), np.array(prepulse_maximums), smooth_parsed_traces

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
    pulse_frames = pulse_frames[1:9]
    prepulse_frames = prepulse_frames[1:9]
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

def split_traces_by_genotype(traces, genotype_csv):
    """Split parsed traces into separate arrays for each unique genotype."""
    genotype_map = pd.read_csv(genotype_csv, header=None)
    unique_genotypes = np.unique(genotype_map.values)
    traces_by_genotype = []
    genotypes = []
    traces=np.array(traces)
    for gene in unique_genotypes:
        if gene == 'None':
            continue
        genotypes.append(gene)

        traces_temp = []
        index = 0

        for row in range(genotype_map.shape[0]):
            for col in range(genotype_map.shape[1]):
                if gene == genotype_map.iloc[row, col]:  
                    traces_temp.append(traces[:, index, :])
                index += 1

        traces_temp = np.array(np.swapaxes(traces_temp, 0, 1))
        traces_by_genotype.append(traces_temp)

    return traces_by_genotype, genotypes

def split_measures_by_genotype(measures, genotype_csv):
    """Split measures into separate arrays for each unique genotype."""
    genotype_map = pd.read_csv(genotype_csv, header=None)
    unique_genotypes = np.unique(genotype_map.values)
    measures_by_genotype = []
    genotypes = []

    for gene in unique_genotypes:
        if gene == 'None':
            continue
        genotypes.append(gene)
        measures_temp = []
        
        index = 0

        for row in range(genotype_map.shape[0]):
            for col in range(genotype_map.shape[1]):
                if gene == genotype_map.iloc[row, col]:  
                    measures_temp.append(measures[index])
                index += 1
                
        measures_by_genotype.append(measures_temp)

    return measures_by_genotype, genotypes


def compute_proportion_responders(norm_pulse_maxs, thresh=0.3):
    # Computes the proportion of fish with maximum values above a specified threshold per trial based on a 2D array of normalized pulse maximums.
    
    mask = norm_pulse_maxs > thresh
    count = np.sum(mask, axis=1)
    total_fish = norm_pulse_maxs.shape[1]
    prop_responders = count / total_fish
    
    return prop_responders

def normalise_traces(traces, fish_maxes=None):
    num_time_points, num_fish, num_trials = traces.shape

    # Create an empty array to store the normalized traces
    normalised_traces = np.empty_like(traces)

    if fish_maxes is None:
        # Compute the maximum motion of each fish from frame 40 to the end of traces across all trials
        print('Finding maximum of all fish after stimulus onset')
        fish_maxes = np.nanmax(traces[40:, :, :], axis=(0, 2))

    # Loop through each trial and time point to normalize the traces
    for trial in range(num_trials):
        for time_point in range(num_time_points):
            trace = traces[time_point, :, trial]
            smoothed_trace = savgol_filter(trace, window_length=5, polyorder=2)
            normalized_trace = smoothed_trace / fish_maxes
            normalized_trace[normalized_trace < 0] = 0  # Clip negative values to 0
            normalised_traces[time_point, :, trial] = normalized_trace

    return normalised_traces, fish_maxes

def collect_PPI_measures(PPIReductionsS, gene_map_pathS):
    # Create empty DataFrames to store the results
    result_df = pd.DataFrame(columns=['Date', 'Genotype', 'PPIReduction'])
    excluded_wells = pd.DataFrame(columns=['Date', 'Genotype', 'PPIReduction'])
    
    # Iterate through each experiment and its corresponding genotype map path
    for experiment, gene_map_path in zip(PPIReductionsS, gene_map_pathS):
        # Extract the date from the genotype map path
        date = gene_map_path.rsplit(maxsplit=1, sep='\\')[1][:6]

        # Read the genotype mapping CSV file into a DataFrame
        genotype_map = pd.read_csv(gene_map_path)

        index=0
        # Iterate through the rows and columns of the genotype map
        for row in range(genotype_map.shape[0]):
            for col in range(genotype_map.shape[1]):
                genotype = genotype_map.iloc[row, col]

                # Extract the corresponding measure from the experiment data (adjust this based on your data structure)
                PPIReduction = experiment[index]  # Replace 'genotype' with the actual key used in your data

                # Check if the genotype is 'None' or 'empty'
                if genotype in ('None', 'empty'):
                    # Append the data to the excluded_wells DataFrame
                    excluded_wells = excluded_wells.append({'Date': date, 'Genotype': genotype, 'PPIReduction': PPIReduction}, ignore_index=True)
                else:
                    # Append the data to the result DataFrame
                    result_df = result_df.append({'Date': date, 'Genotype': genotype, 'PPIReduction': PPIReduction}, ignore_index=True)
                index+=1
    return result_df, excluded_wells

def calculate_mean_without_nans(values):
    # Calculate mean without NaN values
    non_nan_values = [val for val in values if not np.isnan(val)]
    if non_nan_values:
        return np.mean(non_nan_values)
    else:
        return np.nan

def process_PPI_data(PPI_reductions_genes):
    # Apply the function to calculate mean without NaNs
    PPI_reductions_genes['PPIReduction'] = PPI_reductions_genes['PPIReduction'].apply(calculate_mean_without_nans)
    
    # Remove rows with NaN mean values
    PPI_fish_mean = PPI_reductions_genes.dropna(subset=['PPIReduction'])
    
    # Remove rows with mean values outside of the range [-1.5, 1.5]
    PPI_fish_mean = PPI_fish_mean[(PPI_fish_mean['PPIReduction'] >= -1.5) & (PPI_fish_mean['PPIReduction'] <= 1.5)]
    
    return PPI_fish_mean

def calculate_z_scores(dataframe):
    # Create a new DataFrame to store the z-scores
    z_score_df = dataframe.copy()

    # Group the DataFrame by 'date', 'genotype' and compute the mean and standard deviation for 'Scrambled' controls
    control_stats = dataframe[dataframe['Genotype'] == 'Scrambled'].groupby(['Date', 'Genotype']).agg({'PPIReduction': ['mean', 'std']})
    control_stats.columns = ['scrambled_mean', 'scrambled_std']

    # Merge the control statistics with the original DataFrame
    z_score_df = z_score_df.merge(control_stats, on=['Date', 'Genotype'], how='left')

    # Compute z-scores based on the 'Scrambled' controls
    z_score_df['z_score'] = (z_score_df['PPIReduction'] - z_score_df['scrambled_mean']) / z_score_df['scrambled_std']

    # Drop the additional columns used for calculation
    z_score_df.drop(['scrambled_mean', 'scrambled_std'], axis=1, inplace=True)

    return z_score_df

def plot_PPI_swarm_and_stats(dataframe, swarm=True, use_mean=True, use_median=False, use_std=False, use_sem=False, significance_threshold=0.05, order=None):
    # Get unique genotypes
    
    if order != None: 
        genotypes=order
    else:
        genotypes = dataframe['Genotype'].unique()
        
    # Create a DataFrame to store adjusted p-values
    p_values_df = pd.DataFrame(columns=['Genotype', 'Day_p-value', 'Night-p-value'])

    plt.figure(figsize=(15, 5))
    colors = ['black', (10/255, 40/255, 205/255, 1), (175/255, 21/255, 70/255, 1), (0.3, 0.3, 0.3, 1)]

    for i, genotype in enumerate(genotypes):
        genotype_data = dataframe[dataframe['Genotype'] == genotype]

        day_data = genotype_data['PPIReduction'].apply(lambda x: np.nanmean(x))
        scrambled_data = dataframe[dataframe['Genotype'] == 'Scrambled']['PPIReduction'].apply(
            lambda x: np.nanmean(x))

        t_statistic, p_value = ttest_ind(day_data, scrambled_data, equal_var=False)
        
        # Perform multiple comparisons correction
        corrected_p_value = multipletests([p_value], method='holm')[1][0]
        
        # Append p-values to the DataFrame
        p_values_df = p_values_df.append({'Genotype': genotype, 'Day_p-value': corrected_p_value, 'Night-p-value': None}, ignore_index=True)

        # Determine significance and mark column with asterisk if significant
        if corrected_p_value < significance_threshold:
            # plt.text(i, max(dataframe['PPIReduction'].max(), 10), '*', fontsize=16, ha='center', va='center', fontweight='bold')
            color_index = 1 if np.nanmean(day_data) < np.nanmean(scrambled_data) else 2
        else:
            color_index = 3
            
        # Plot the swarm or stripplot
        if swarm:
            sns.swarmplot(x=[i] * len(day_data), y=day_data, color=colors[color_index], alpha=0.2)
        else:
            sns.stripplot(x=[i] * len(day_data), y=day_data, color=colors[color_index], size=3, alpha=0.2)

        if use_mean:
            measure = np.nanmean(day_data)
        elif use_median:
            measure = np.nanmedian(day_data)

        if use_std:
            error = np.nanstd(day_data)
        elif use_sem:
            error = np.nanstd(day_data) / np.sqrt(len(day_data))
        else:
            ci = np.percentile(day_data, [25, 75])
            error = (ci[1] - ci[0]) / 2

        # Plot mean or median as a dot with colored based on significance
        plt.scatter(i, measure, color=colors[color_index], marker='o', s=10)

        # Plot error bars
        plt.errorbar(i, measure, yerr=error, color=colors[color_index], linewidth=3)

    plt.xlabel('Genotype')
    plt.ylabel('PPI Reduction - Day')
    plt.title('PPI Reduction - Day')
    plt.xticks(np.arange(len(genotypes)), genotypes, rotation=45)
    
    plt.tight_layout()

    return p_values_df

def plot_PPI_swarm_and_stats1(dataframe, swarm=True, use_mean=True, use_median=False, use_std=False, use_sem=False, significance_threshold=0.05,order=None):
    # Get unique genotypes
    if order != None: 
        genotypes=order
    else:
        genotypes = dataframe['Genotype'].unique()

    # Create a DataFrame to store adjusted p-values
    p_values_df = pd.DataFrame(columns=['Genotype', 'p-value'])

    plt.figure(figsize=(15, 5))
    colors = ['black', (10/255, 40/255, 205/255, 1), (175/255, 21/255, 70/255, 1), (0.3, 0.3, 0.3, 1)]

    for i, genotype in enumerate(genotypes):
        genotype_data = dataframe[dataframe['Genotype'] == genotype]

        day_data = genotype_data['PPIReduction'].apply(lambda x: np.nanmean(x))
        scrambled_data = dataframe[dataframe['Genotype'] == 'Scrambled']['PPIReduction'].apply(
            lambda x: np.nanmean(x))

        t_statistic, p_value = ttest_ind(day_data, scrambled_data, equal_var=False)
        
        # Perform multiple comparisons correction
        corrected_p_value = multipletests([p_value], method='holm')[1][0]
        
        # Append p-values to the DataFrame
        p_values_df = p_values_df.append({'Genotype': genotype, 'p-value': corrected_p_value}, ignore_index=True)

        # Determine significance and mark column with asterisk if significant
        if corrected_p_value < significance_threshold:
            # plt.text(i, max(dataframe['PPIReduction'].max(), 10), '*', fontsize=16, ha='center', va='center', fontweight='bold')
            if np.nanmean(day_data) < np.nanmean(scrambled_data):
                color_index = 1 
            else: 
                color_index = 2
        else:
            color_index = 3
        if genotype == 'Scrambled':
            color_index=0
            
        # Plot the swarm or stripplot
        if swarm:
            sns.swarmplot(x=[i] * len(day_data), y=day_data, color='black', size=3, alpha=0.2)
        else:
            sns.stripplot(x=[i] * len(day_data), y=day_data, color='black', size=3, alpha=0.1)

        if use_mean:
            measure = np.nanmean(day_data)
        elif use_median:
            measure = np.nanmedian(day_data)

        if use_std:
            error = np.nanstd(day_data)
        elif use_sem:
            error = np.nanstd(day_data) / np.sqrt(3)
        else:
            ci = np.percentile(day_data, [25, 75])
            error = (ci[1] - ci[0]) / 2

        # Plot mean or median as a dot with colored based on significance
        # plt.scatter(i, measure, color=colors[color_index], marker='o', s=20)
        plt.plot([i - 0.15, i + 0.15], [measure , measure ], color=colors[color_index], linewidth=2)  # Measure horizontal line
        # Plot error bars as smaller horizontal lines
        plt.plot([i,i],[measure-error,measure+error], color=colors[color_index], linewidth=2)  # Vertical line
        plt.plot([i - 0.1, i + 0.1], [measure - error, measure - error], color=colors[color_index], linewidth=2)  # Lower horizontal line
        plt.plot([i - 0.1, i + 0.1], [measure + error, measure + error], color=colors[color_index], linewidth=2)  # Upper horizontal line

    plt.xlabel('Genotype')
    plt.ylabel('PPI Reduction Fraction')
    plt.title('PPI Reduction')
    plt.xticks(np.arange(len(genotypes)), genotypes, rotation=45)
    
    plt.tight_layout()

    return p_values_df


#%% Script
if __name__ == "__main__":
    
    pulse_threshold=200 # arbitrary threshold to determine if a response to a pulse happened or not to compute the average pulse amplitude
    frame_rate=100
    prepulse_delay_frames=30
    ##################################
    plot=False
    trackXY=False
    ##################################
    fullRange=True
    frame_range=30
    ##################################
    # folderListFile=r'D:\dataToTrack\Habituation\FolderLists\Habituation_1000K.txt'
    folderListFile='S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/PPI/Data/FolderLists/PPI_Tracking_230928_all_maxAmp.txt'
    # folderListFile='S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/PPI/Data/FolderLists/PPI_Tracking_230928_all_minAmp.txt'
    data_path,folderNames=PPIU.read_folder_list(folderListFile)
    n=1
    N=len(folderNames)
    
    pulseTracesS,prepulseTracesS=[],[]
    pulseTraces_smoothS,prepulseTraces_smoothS = [],[]
    pulseTraces_normS,prepulseTraces_normS=[],[]
    pulseMeasuresS,prepulseMeasuresS, prepulsepreMeasuresS=[],[],[]
    pulseMeasures_normS,prepulseMeasures_normS, prepulsepreMeasures_normS=[],[],[]
    PPIReductionsS = []
    gene_map_pathS,expDateS = [], []
    # pulse_prop_respondersS,prepulse_prop_respondersS = [],[]
    for folder in folderNames:
        print(f'Processing video {n} out of {N} in folder {folder}')
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
        gene_map_pathS.append(gene_map_path)

        # Grab traces and time index (convert to s)
        df=pd.read_csv(traces_path,header=None)
        traces=np.array(df)
        time_s = [(f / frame_rate) for f in traces[:,0]]
        traces=traces[:,1:]
        df=df.iloc[:,1:]
        
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
        # pulseMeasuresS.append(pulse_response_integrals)
        
        # Compute the maximums of each pulse from the smoothed trace
        pulse_maximums,_,pulse_traces_smooth=compute_smoothed_maximum(time_seg,pulse_traces,fullRange=fullRange,frame_range=frame_range, prepulse_delay_frames=prepulse_delay_frames)
        
        # Normalise traces by the maximums for each fish across trials
        masked_pulse_maximums = np.copy(pulse_maximums)
        masked_pulse_maximums[masked_pulse_maximums < 200] = np.nan
        fish_maxes=np.nanmean(masked_pulse_maximums,axis=1)
        pulse_traces_norm, _ = normalise_traces(pulse_traces_smooth, fish_maxes=fish_maxes)
        
        # Collect data for this folder
        pulseTracesS.append(pulse_traces)
        pulseTraces_smoothS.append(pulse_traces_smooth)
        pulseMeasuresS.append(pulse_maximums)
        pulseTraces_normS.append(pulse_traces_norm)
        
        # average traces across trials and save for this folder
        pulse_traces_norm_av=np.mean(pulse_traces_norm,axis=2)
        dfSave = pd.DataFrame(np.array(pulse_traces_norm_av))
        dfSave.to_csv(folder+r'/perfish_average_pulseTraces_norm.csv')
        
        # if this is a prepulse experiment, there will also be prpepulse frames
        if len(prepulse_frames)>1:
            # extract prepulse trial traces
            _,prepulse_traces=segment_habituation(df, prepulse_frames)
            
            # Compute maximums at the pulse from smoothed traces
            prepulse_pulse_maximums,prepulse_prepulse_maximums,prepulse_traces_smooth=compute_smoothed_maximum(time_seg,prepulse_traces,fullRange=fullRange,frame_range=frame_range, prepulse_delay_frames=prepulse_delay_frames)
            
            prepulse_traces_norm, _ = normalise_traces(prepulse_traces_smooth, fish_maxes=fish_maxes)
            # prepulse_pulse_proportion_responders=compute_proportion_responders(prepulse_pulse_maximums_norm, thresh=threshold) 
            # prepulse_prepulse_proportion_responders=compute_proportion_responders(prepulse_prepulse_maximums_norm, thresh=threshold) 
            
            prepulseTracesS.append(prepulse_traces)
            prepulseTraces_smoothS.append(prepulse_traces_smooth)
            prepulseTraces_normS.append(prepulse_traces_norm)
            
            prepulseMeasuresS.append(prepulse_pulse_maximums)
            prepulsepreMeasuresS.append(prepulse_prepulse_maximums)
            # prepulseMeasures_normS.append(prepulse_pulse_maximums_norm)
            # prepulsepreMeasures_normS.append(prepulse_prepulse_maximums_norm)
            
            prepulse_traces_norm_av=np.mean(prepulse_traces_norm,axis=2)
            dfSave = pd.DataFrame(np.array(prepulse_traces_norm_av))
            dfSave.to_csv(folder+r'/perfish_average_prepulseTraces_norm.csv')
            
            # Proportion Difference for every trial compared to average of all pulse trials (where the fish responded over low threshold)
            PPIReductions=[]
            for nn,fish_max in enumerate(fish_maxes):
                prepulse_amps=prepulse_pulse_maximums[nn]
                PPIreduction_temp=[]
                for prepulse in prepulse_amps:
                    PPIreduction_temp.append(1-(prepulse)/fish_max)
                PPIReductions.append(PPIreduction_temp)
                
            dfSave = pd.DataFrame(PPIReductions)
            dfSave.to_csv(folder+r'/PPI_average_reduction_per_fish.csv')
            print(f'Saved PPI reduction data and normalised traces at {folder}')
        PPIReductionsS.append(PPIReductions)
            
#%% Remove excluded experiments. No internal control, synch with stimulus lost, too many non responders
keep_indexes=[True,True,True,True,True,False,True,True,False,False]
PPIReductionsS=PPIReductionsS[1:]
pulseMeasuresS=pulseMeasuresS[1:]
prepulseMeasuresS=prepulseMeasuresS[1:]
prepulsepreMeasuresS=prepulsepreMeasuresS[1:]
pulseTracesS=pulseTracesS[1:]
prepulseTracesS=prepulseTracesS[1:]
folderNames_temp=[]
folderNames=folderNames[1:]
for n,folder in enumerate(folderNames):
    if keep_indexes[n]:
        folderNames_temp.append(folder)
folderNames=folderNames_temp

PPIReductionsS=np.array(PPIReductionsS)[keep_indexes].tolist()
pulseMeasuresS=np.array(pulseMeasuresS)[keep_indexes].tolist()
prepulseMeasuresS=np.array(prepulseMeasuresS)[keep_indexes].tolist()
prepulsepreMeasuresS=np.array(prepulsepreMeasuresS)[keep_indexes].tolist()
pulseTracesS=np.array(pulseTracesS)[keep_indexes].tolist()
prepulseTracesS=np.array(prepulseTracesS)[keep_indexes].tolist()

#%% Debug and analyse
pPAs=[.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1]
PAs=[x / 10 for x in pPAs]
genes=[]
genes.append(r'Scrambled')
genes.append(r'trio')
# genes.append(r'gria3')
genes.append(r'grin2a')
genes.append(r'cacna1g')
genes.append(r'sp4') 
genes.append(r'xpo7')
# genes.append(r'akap11')    
# genes.append(r'herc1')  
genes.append(r'hcn4')
# genes.append(r'nr3c2')


# Collect measures into dataframe (would be better to do this in the above cell but this is a quickish fix)
# seperate by genotype
PPIReductions_genes, excluded_wells = collect_PPI_measures(PPIReductionsS, gene_map_pathS)
# Take the mean per fish
PPI_fish_mean = process_PPI_data(PPIReductions_genes)

# zscore all fish against scrambled fish in the same experiment
# z_score_df = calculate_z_scores(PPI_fish_mean)
p_values = plot_PPI_swarm_and_stats1(PPI_fish_mean, swarm=True, use_median=False, use_sem=True, significance_threshold=0.05, order=genes)

#%% the rest
# pulseMeans, prepulseMeans, prePrepulseMeans = [], [], []
import matplotlib.cm as cm
for n, folder in enumerate(folderNames):
    plt.close('all')
    folder=folderNames[n]   
    saveFolder=folder+'/Figures'
    SCZU.cycleMkDir(saveFolder)
    csvs=glob.glob(folder+r'/*.csv')
    parameter_path=[item for item in csvs if "parameters" in item][0]
    left=[item for item in csvs if "parameters" not in item]
    norm_traces_path=[item for item in left if "motionTraces" in item][0]
    left=[item for item in left if "motionTraces" not in item]
    gene_map_path=[item for item in left if "GenotypeMap" in item][0]
    stim_path=[item for item in left if "GenotypeMap" not in item][0]
    # stim_path=[item for item in csvs if "parameters" not in item][0]
    # find the stimulus times and amplitudes
    csv=pd.read_csv(stim_path,header=None)
    amplitudes=list(set(csv[2].array))   
    pulse_frames,prepulse_frames=separate_pulse_prepulse_frames(stim_path)
    
    # Grab traces
    df=pd.read_csv(traces_path,header=None)
    traces=np.array(df)
    
    # traces=traces[:,1:]
    pulse_traces=pulseTracesS[n]#[:,:,1:]
    prepulse_traces=prepulseTracesS[n]#[:,:,1:]
    
    pulse_measures=pulseMeasuresS[n]#[:,1:]
    prepulse_measures=prepulseMeasuresS[n]#[:,1:]
    prepulse_premeasures=prepulsepreMeasuresS[n]#[1:,1:]
    
    
    ######################### SPLIT INTO GENOTYPES ###########################
    pulse_gene_traces,genotypes = split_traces_by_genotype(pulse_traces,gene_map_path)
    prepulse_gene_traces,_ = split_traces_by_genotype(prepulse_traces,gene_map_path)
    pulse_gene_measures,_ = split_measures_by_genotype(pulse_measures,gene_map_path)
    prepulse_gene_measures,_ = split_measures_by_genotype(prepulse_measures,gene_map_path)
   
    for nG, genotype in enumerate(genotypes):
        # grab traces for this genotype
        pulse_traces=np.array(pulse_gene_traces[nG])
        prepulse_traces=np.array(prepulse_gene_traces[nG])
        
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
        
        # Plot distributions of the measures for this experiment (amplitudes usually)
        pulse_measures=np.array(pulse_gene_measures[nG])
        prepulse_measures=np.array(prepulse_gene_measures[nG])
        prepulse_premeasures=np.array(prepulse_premeasures[nG])
        
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
        
        # Plot the trial traces
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
        plt.savefig(saveFolder+'/TRYPlate_motion_per_trial_P' + str(PAs[n]) + '_pP' + str(pPAs[n]) + '_' + str(genotype) + '%.png', dpi=600)
        
        plt.figure()
        # Number of trials
        num_trials = len(pulseMeanPerTrial[0])
        prepulse_color = 'black'

        # Create a color array that transitions from blue to red
        cmap = cm.coolwarm
        colors = [cmap(i / (num_trials - 1)) for i in range(num_trials)]
        
        # Plot the traces
        for i in range(num_trials):
            plt.plot(time_seg, pulseMeanPerTrial[:,i], color=colors[i],label=f'Pulse Trial {i}')

        for i in range(num_trials):
            alpha = 1.0 - i / num_trials
            plt.plot(time_seg, prepulseMeanPerTrial[:,i], color=prepulse_color, alpha=alpha, label=f'Prepulse Trial {i}')

        
        plt.xlabel('Time relative to pulse (s)')
        plt.ylabel('Motion (AU)')
        plt.savefig(saveFolder+'/HOTCOLDPlate_MeanMotion_P' + str(PAs[n]) + '_pP' + str(pPAs[n]) + '_' + str(genotype) + '%.png', dpi=600)

        print(f'Figures saved in {saveFolder}')
        
# Plot PPI reductions of all genotypes together
 
 
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
