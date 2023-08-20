# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 03:34:29 2023

@author: Tom
"""

lib_path = r'D:\Tom\Github\SCZ_Fish\libs'
import sys
sys.path.append(lib_path)

import cv2
import imageio
import numpy as np
import SCZ_utilities as SCZU
import SCZ_video as SCZV
import PPI_utilities as PPIU

def compute_background(vid, num_frames=50):
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_indices = np.linspace(0, int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, num_frames, dtype=int)
    frames = []

    while len(frames) < num_frames:
        ret, frame = vid.read()
        if vid.get(cv2.CAP_PROP_POS_FRAMES) - 1 in frame_indices:
            frames.append(frame)

    background = np.mean(frames, axis=0).astype(np.uint8)
    return background

def test_roi_grid(im, rois):
    for x, y, w, h in rois:
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the frame with ROIs
    cv2.imshow("ROI Grid Test", im)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

def generate_motion_trace(vid, rois, max_frames=-1, diffThresh=5, num_rois=None, start_roi=None, makeMovie=True):
    
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-1
    if max_frames > 0 and max_frames < num_frames:
        num_frames = max_frames
    else:
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-1

    if num_rois is None:
        num_rois = len(rois)

    if start_roi is None:
        start_roi = 0

    motion_trace_array = np.zeros((num_frames, num_rois))
    brightness_array = np.zeros((num_frames, num_rois))
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, prev_frame = vid.read()

    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diff_frame_movie=[]
    for i in range(num_frames):
        ret, frame = vid.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_frame = cv2.absdiff(gray_frame, prev_frame)
        diff_frame[diff_frame < diffThresh] = 0
        if makeMovie:
            diff_frame_movie.append(diff_frame)
        else:
            diff_frame_movie=-1
        # print('looping through ROIs this frame')
        for j in range(num_rois):
            x, y, w, h = rois[j]
            m = diff_frame[y:y + h, x:x + w]
            # print(f'ROI {j}, Frame {i}: {np.sum(m)}')
            # if np.sum(m)>1500:
                # asdasd=2
                # print(f"ROI : {j} / Frame : {i}")
            motion_trace_array[i, j] = np.sum(m)
            # debug
            brightness_array[i,j]=np.sum(gray_frame)
            ###
        prev_frame = np.copy(gray_frame)
        if (i + 1) % 1000 == 0:
            percentage_completion = ((i + 1) / num_frames) * 100
            print(f"Processing {percentage_completion:.0f}% complete...")

    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return brightness_array,motion_trace_array, diff_frame_movie

# Debug function to make specific ROI diff_frame
def diffFrameROI(vid, frame,rois,roiNum,diffThresh=7,plot=True):
    [x,y,w,h]=rois[roiNum]
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame-1)
    ret,prev=vid.read()
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev=prev[y:y + h, x:x + w]
    ret,im=vid.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im=im[y:y + h, x:x + w]
    diff_frame_ROI = cv2.absdiff(im,prev)
    diff_frame_ROI[diff_frame_ROI < diffThresh] = 0
    if plot:
        plt.figure('im1');plt.imshow(im)
        plt.figure('prev1');plt.imshow(prev)
        plt.figure('diff1');plt.imshow(diff_frame_ROI)
    return np.sum(diff_frame_ROI),diff_frame_ROI, im, prev

    #  In case the opencv library isn't working to save video
def save_vid_from_list_imageio(image_frames, fileName='diff_movie.mp4', frame_rate = 100):
    # Convert grayscale frames to 3-channel BGR format
    frames_bgr = [np.stack((frame,) * 3, axis=-1) for frame in image_frames]

    # Create an imageio.get_writer() object to save the video
    writer = imageio.get_writer(fileName, fps=frame_rate)

    # Loop through each image frame and add it to the video
    for frame_bgr in frames_bgr:
        writer.append_data(frame_bgr)

    # Close the writer to finalize the video
    writer.close()

    return fileName

def save_vid_from_list_cv2(image_frames, fileName='diff_movie.mp4', frame_rate = 100):
    # Get the height and width of the first frame
    height, width = image_frames[0].shape
    norm_max=np.max(image_frames)
    image_frames = image_frames/np.max(norm_max)
    image_frames = np.ubyte(image_frames*255)
    
    # Enhance Contrast (Histogram Equalize)
    equ = cv2.equalizeHist(image_frames)
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(fileName, fourcc, frame_rate, (width, height))
    
    # Loop through each grayscale image frame and write it to the video
    for frame in equ:
        # Convert single-channel grayscale image to 3-channel (BGR) format
        frame_bgr = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
        output_video.write(frame_bgr)

    # Release the VideoWriter object
    output_video.release()

    return fileName

# Remove weird spikes every 12th frame by interpolating across the adjacent points.
# Where are these coming from (8.33Hz precise shot noise)!? 
# Update: 08/23 this can also be overcome by increasing individual pixel threshold for detection if signal is high enough
def interp_series_every_n_points(trace, interval=12):
    indices = np.arange(0, len(trace), interval)
    interpTrace = np.interp(np.arange(len(trace)), indices, trace[indices])
    return interpTrace

#%%    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import glob
    import pandas as pd    
    
    plot=False
    makeMotionTrace = True
    trackXY=False
    makeMovie=False
    folderListFile=r'D:\dataToTrack\Habituation\FolderLists\Habituation_1000K.txt'
    
    data_path,folderNames=PPIU.read_folder_list(folderListFile)
    # folderNames=[r'D:\dataToTrack\Habituation\Plate_1\Amp_1500000\Exp1_ISI_1s']
    
    n=1
    N=len(folderNames)
    for folder in folderNames:
        print(f'Processing video {n} out of {N}')
        n+=1
        movie_path=glob.glob(folder+r'/*.avi')[0]
        csvs=glob.glob(folder+r'/*.csv')
        parameter_path=[item for item in csvs if "params" in item][0]
        stim_path=[item for item in csvs if "params" not in item][0]
        
        # ROI_TL=[16,36,79,74]
        # ROI_BR=[968,649,85,86]
        ROI_TL,ROI_BR=PPIU.load_TLBR_ROIs(folder)
        
        vid = cv2.VideoCapture(movie_path) 
        max_frames=12000
        
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret,first_frame=vid.read()
        rois=PPIU.create_roi_grid_from_corners(ROI_TL,ROI_BR, num_rows=8, num_cols=12)
        test_roi_grid(first_frame, rois)
        
        numFrames=int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        numROIs=len(rois)
        
        if makeMotionTrace:
            print(f'Computing traces from {numROIs} ROIs, video is {numFrames} frames long')
            brightness,traces,diff_frame_movie = generate_motion_trace(vid, rois, diffThresh=12, max_frames=max_frames, makeMovie=makeMovie)
            print('Saving traces as ' + data_path + r'/motionTraces.csv')
            df = pd.DataFrame(np.array(traces))
            df.to_csv(data_path+r'/motionTraces.csv')
            
        if makeMovie:
            print('Saving diff video')
            save_vid_from_list_cv2(diff_frame_movie,fileName='diff_movie.mp4')
            
        if trackXY:
            print('Tracking fishes... May take some time...')
            roi_array=np.array(rois)
            trackingFolder=data_path+r'/Tracking'
            SCZU.cycleMkDir(trackingFolder)
            FigFolder=trackingFolder+r'/Figures'
            SCZU.cycleMkDir(FigFolder)
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS, distS = SCZV.PPI_96well_fishtracking(vid, FigFolder, roi_array, report=True, status=[1,1,1,1,1,1], endMins=-1, skipFirst=0)
            
            for i in range(0,fxS.shape[1]):
                filename = trackingFolder + r'/tracking' + str(i+1) + r'.npz'
                fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
                print(f'Saving tracking at {filename}')
                np.savez(filename, tracking=fish.T)
        
        # Close Plots
        plt.close('all')
        
        # Close video
        vid.release()
    print('FIN')
    