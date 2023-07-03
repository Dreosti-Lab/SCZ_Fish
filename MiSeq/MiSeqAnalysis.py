# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:31:32 2023

@author: Tom
"""

import zipfile
import pandas as pd
import numpy as np
import glob
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'D:\Tom\Github\SCZ_Fish\libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import SCZ_utilities as SCZU
# Collection of functions and scripts to analyse MiSeq data from Schizophrenia 


def computeKO_from_Crispresso_single(CrispressoFolder,threshold=500,report=False):
    """
    Compute KO alleles from Crispresso single output.

    Args:
        CrispressoFolder (str): Path to the Crispresso output folder (zip file).
        threshold (int): Threshold for the sum of '#Reads' column (default: 500).
        report (bool): Whether to print the results (default: False).

    Returns:
        tuple: A tuple containing the proportion of wildtype alleles and proportion of KO alleles.
               if number of reads is below threshold, returns (-1,-1)
        
    Description:
        Takes a zipped folder downloaded from Crispresso and processes the allele frequency table 
        to compute the proportion of reads that are KO vs wildtype in a single alignment. 
        KO is defined as a frame shift insertion or deletion (not divisible by 3) or at least 40 bp
        insertion or deletion. 
"""
    prop_ko=0.0
    temp_unzip = CrispressoFolder.rsplit(sep='\\',maxsplit=1)[0]+'\\'
    
    # Open the zip file
    with zipfile.ZipFile(CrispressoFolder, 'r') as zip_ref:
        freqTable = [file for file in zip_ref.namelist() if file.endswith('.txt') and file.startswith('Alleles_frequency_table')][0]
        # Extract the file to a temporary location
        zip_ref.extract(freqTable, temp_unzip)      
        df = pd.read_csv(temp_unzip + freqTable, delimiter='\t')
        
        # Remove rows with < 1% in '%Reads' column
        df = df[df['%Reads'] >= 1]
    
        # Sum the '#Reads' column and continue only if greater than threshold
        total_reads = df['#Reads'].sum()
        if total_reads > threshold:
            # Check for 'Unedited' column and compute proportion of wildtype alleles
            if 'Unedited' in df.columns:
                unedited_mask = df['Unedited'] == True
                prop_wildtype = df.loc[unedited_mask, '%Reads'].sum()
    
            # Loop through remaining rows and compute 'PropKO'
            mut_allele_count=0
            for _, row in df.iterrows():
                num_insertions = row['n_inserted']
                num_deletions = row['n_deleted']
    
                # Check conditions for 'PropKO'
                if (abs(num_insertions - num_deletions) % 3 != 0) or (abs(num_insertions) > 20) or (abs(num_deletions) > 20):
                    prop_ko += row['%Reads']
                    mut_allele_count += 1
        else:
            if report:
                print("Not enough reads! Threshold set to " + str(threshold))
            return -1,-1
        if report:
        # Print the results
            print("Proportion of Wildtype Alleles:", prop_wildtype)
            print("Proportion of KO Alleles:", prop_ko)
            print("Proportion of edited, but not KO:", np.abs(100- (prop_ko + prop_wildtype)))
    return prop_wildtype, prop_ko

def computeKO_from_Crispresso_singleFolder(CrispressoFolder,threshold=500,report=False):
    """
    Compute KO alleles from Crispresso single output.

    Args:
        CrispressoFolder (str): Path to the Crispresso output folder (folder).
        threshold (int): Threshold for the sum of '#Reads' column (default: 500).
        report (bool): Whether to print the results (default: False).

    Returns:
        tuple: A tuple containing the proportion of wildtype alleles and proportion of KO alleles.
               if number of reads is below threshold, returns (-1,-1)
        
    Description:
        Takes a unzipped folder downloaded from Crispresso and processes the allele frequency table 
        to compute the proportion of reads that are KO vs wildtype in a single alignment. 
        KO is defined as a frame shift insertion or deletion (not divisible by 3) or at least 40 bp
        insertion or deletion. 
"""
    prop_ko=0.0
    freqTable=glob.glob(CrispressoFolder+'\\Alleles_frequency_table*.txt')[0]  
    df = pd.read_csv(freqTable, delimiter='\t')
    
    # Remove rows with < 1% in '%Reads' column
    df = df[df['%Reads'] >= 1]

    # Sum the '#Reads' column and continue only if greater than threshold
    total_reads = df['#Reads'].sum()
    if total_reads > threshold:
        # Check for 'Unedited' column and compute proportion of wildtype alleles
        if 'Unedited' in df.columns:
            unedited_mask = df['Unedited'] == True
            prop_wildtype = df.loc[unedited_mask, '%Reads'].sum()
        
            # Loop through remaining rows and compute 'PropKO'
            mut_allele_count=0
            for _, row in df.iterrows():
                num_insertions = row['n_inserted']
                num_deletions = row['n_deleted']

            # Check conditions for 'PropKO'
                if (abs(num_insertions - num_deletions) % 3 != 0) or (abs(num_insertions) > 20) or (abs(num_deletions) > 20):
                    prop_ko += row['%Reads']
                    mut_allele_count += 1
    else:
        if report:
            print("Not enough reads! Threshold set to " + str(threshold))
        return -1,-1
    if report:
    # Print the results
        print("Proportion of Wildtype Alleles:", prop_wildtype)
        print("Proportion of KO Alleles:", prop_ko)
        print("Proportion of edited, but not KO:", np.abs(100- (prop_ko + prop_wildtype)))
    return prop_wildtype, prop_ko


#%% script

folderListFile='S:\WIBR_Dreosti_Lab\Tom\Crispr_Project\Miseq\MiSeq_data\FolderLists\RyanMiseqJun22.txt'
founders=False

if founders:
    folderPath, genotypes, fishNums, folderNames = SCZU.read_folder_list_founders(folderListFile) 
    genoSet=set(genotypes)
    saveName=folderPath+'\F1_FounderReport.csv'
else:
    folderPath, genotypes, fishNums, folderNames = SCZU.read_folder_list_MiSeq(folderListFile)
    saveName=folderPath+'\MiSeq_Report.csv'
    
# Loop through fish
data = {
    'Genotype': [],
    'FishNum': [],
    'GuideNum': [],
    'Prop_WT': [],
    'Prop_KO': []
}
for idx,folder in enumerate(folderNames):
    # How many guides in this folder?
    zipFlag=True
    guideZips=glob.glob(folder+'\\*.zip')
    if len(guideZips)==0:
        guideZips=glob.glob(folder+'\\CRISPResso_Report*')
        zipFlag=False
    for idg,guideZip in enumerate(guideZips):
        # loop through guides
        if zipFlag:
            prop_wildtype, prop_ko = computeKO_from_Crispresso_single(guideZip)
        else:
            prop_wildtype, prop_ko = computeKO_from_Crispresso_singleFolder(guideZip)
        
        # Collect in dataframe
        data['Genotype'].append(str(genotypes[idx]))
        data['FishNum'].append(int(fishNums[idx]))
        data['GuideNum'].append(int(idg))
        data['Prop_WT'].append(float(prop_wildtype))
        data['Prop_KO'].append(float(prop_ko))
df=pd.DataFrame(data)        
df.to_csv(saveName, index=False)
print('Finished, saving results to ' + str(saveName))
# CrispressoFolder = 'S:\WIBR_Dreosti_Lab\Tom\Crispr_Project\Miseq\MiSeq_data\F1_RyanMiseqMay23\CrispresoResults\Founders\HCN4\Fish9\CRISPResso_Report_F04_HCN4-B.zip'#




