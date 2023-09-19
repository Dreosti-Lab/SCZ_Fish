# -*- coding: utf-8 -*-
"""
Compare summaries of analyzed social preference experiments

@author: Tom Ryan, UCL (Dreosti-Group) 
"""
# -----------------------------------------------------------------------------
lib_path = r'D:\Tom\Github\SCZ_Fish\libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)
# -----------------------------------------------------------------------------
#%% Collect and save data
# Set Library Paths
import sys
sys.path.append(lib_path)

# import custom libraries
import SCZ_analysis as SCZA
# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob

# Set "Base Path" for this analysis session
base_path=r'D:/dataToTrack/'
base_path = base_path + r'/Final_Social_SCZ_Analysis/' 
date = '230918'
folderListFile = r'S:\WIBR_Dreosti_Lab\Tom\Crispr_Project\Behavior\Social\FolderLists\All_cohorts_FINAL.txt'
# OR set path to saved dataframe (ALL)
path=None
# path=r'C:/Users/Tom/'
saveDataframe=True
turnThresh=9.75 # degrees to consider a "turn" vs forward swim
# plt.close('all')
if path is not None:
    print('Loading from file...')
    dfAll = pd.read_pickle(path + r"All_SCZ_Fish_" + date + ".pkl")
    dfZ = pd.read_pickle(path + r"All_SCZ_Fish_Zscore_" + date + ".pkl")
    dfIQR = pd.read_pickle(path + r"All_SCZ_Fish_IQR_" + date + ".pkl")
    bouts_df = pd.read_pickle(path + "All_SCZ_Fish_Bouts_" + date)
    print('done loading')
else:

# =============================================================================
    # _, _, folderNames, _, _ = SCZU.read_folder_list1(folderListFile)
    # genes=[r'Scrambled']
    # for idx,folder in enumerate(folderNames):
        
    #     gene=folder.rsplit(sep='\\',maxsplit=3)[1]
    #     if gene not in genes and gene != 'Scrambled':
    #         genes.append(gene)
# OR-OR-OR-OR-OR---------------------------------------------------------------       
    # If you wish to manually control the order of genes, express them here
    genes=[]
    genes.append(r'Scrambled')
    genes.append(r'trio')
    genes.append(r'xpo7')
    genes.append(r'sp4')
    genes.append(r'gria3')
    genes.append(r'grin2a')
    genes.append(r'cacna1g')
    genes.append(r'hcn4')
    genes.append(r'nr3c2')
    genes.append(r'akap11')    
    genes.append(r'herc1')    
    
#==============================================================================
    conditionNames=genes
    analysisFolders=[]
    # Set analysis folder and label for experiment/condition A
    for gene in genes:
        analysisFolders.append(base_path + r'/' + gene)
    
    VPI_Thresh=0.9
    # Summary Containers
    VPI_NS_summary = []
    VPI_S_summary = []
    BPS_NS_summary = []
    BPS_S_summary = []
    Distance_NS_summary = []
    Distance_S_summary = []
    Freezes_NS_summary = []
    Freezes_S_summary = []
    Long_Freezes_NS_summary = []
    Long_Freezes_S_summary = []
    Percent_Moving_NS_summary = []
    Percent_Moving_S_summary = []
    midCrossings_NS_summary = []
    midCrossings_S_summary = []
    propTurns_NS_summary = []
    propTurns_S_summary = []
    boutAbsAngleMean_NS_summary = []
    boutAbsAngleMean_S_summary = []
    boutDistMean_NS_summary = []
    boutDistMean_S_summary = []
    boutDists_NS_summary = []
    boutDists_S_summary = []
    boutAngles_NS_summary = []
    boutAngles_S_summary = []
    
    # Go through each condition (analysis folder)
    count_removals=[]
    AllFish=[]
    AllFish_scram=[]
    AllFish_cond=[]
    for i, analysisFolder in enumerate(analysisFolders):
        
        # Freeze time threshold
        # freeze_threshold = 5*FPS
        # long_freeze_threshold = 1*60*FPS #More than 1 minutes
        
        # Find all the npz files saved for each group and fish with all the information
        npzFiles = glob.glob(analysisFolder+'/*.npz')
        
        # Calculate how many files
        numFiles = np.size(npzFiles, 0)
    
        # Allocate space for summary data
        # VPI_NS_ALL = np.zeros(numFiles)
        # VPI_S_ALL = np.zeros(numFiles)        
        # BPS_NS_ALL = np.zeros(numFiles)
        # BPS_S_ALL = np.zeros(numFiles)
        # Distance_NS_ALL = np.zeros(numFiles)
        # Distance_S_ALL = np.zeros(numFiles)    
        # Freezes_NS_ALL = np.zeros(numFiles)
        # Freezes_S_ALL = np.zeros(numFiles)
        # Percent_Moving_NS_ALL = np.zeros(numFiles)
        # Percent_Moving_S_ALL = np.zeros(numFiles)
        # Long_Freezes_NS_ALL = np.zeros(numFiles)
        # Long_Freezes_S_ALL = np.zeros(numFiles)
        
        VPI_NS_ALL = []
        VPI_S_ALL = []
        BPS_NS_ALL = []
        BPS_S_ALL = []
        Distance_NS_ALL = []
        Distance_S_ALL = []
        Freezes_NS_ALL = []
        Freezes_S_ALL = []
        Percent_Moving_NS_ALL = []
        Percent_Moving_S_ALL = []
        Long_Freezes_NS_ALL = []
        Long_Freezes_S_ALL = []
        midCrossings_NS_ALL = []
        midCrossings_S_ALL = []
        boutAbsAngleMean_NS_ALL = []
        boutAbsAngleMean_S_ALL = []
        boutDistMean_NS_ALL = []
        boutDistMean_S_ALL = []
        propTurns_NS_ALL = []
        propTurns_S_ALL = []
        boutAngles_NS_ALL = []
        boutAngles_S_ALL = []
        boutDists_NS_ALL = []
        boutDists_S_ALL = []
        # Go through all the files contained in the analysis folder
        
        print('Going through ' + str(len(npzFiles)) + ' files for ' + str(genes[i]))
        count=0
        for f, filename in enumerate(npzFiles):
            
            # Load each npz file
            dataobject = np.load(filename)
            
            # Extract from the npz file
            VPI_NS = dataobject['VPI_NS']    
            VPI_S = dataobject['VPI_S']   
            BPS_NS = dataobject['BPS_NS']   
            BPS_S = dataobject['BPS_S']
            Distance_NS = dataobject['Distance_NS']   
            Distance_S = dataobject['Distance_S']   
            Pauses_NS = dataobject['Pauses_NS']   
            Pauses_S = dataobject['Pauses_S']
            Percent_Moving_NS = dataobject['Percent_Moving_NS']   
            Percent_Moving_S = dataobject['Percent_Moving_S']
            Freezes_NS = dataobject['Freezes_NS']
            Freezes_S = dataobject['Freezes_S']
            Long_Freezes_NS = dataobject['Long_Freezes_NS']
            Long_Freezes_S = dataobject['Long_Freezes_NS']
            # Count Freezes
            # Freezes_S = np.array(np.sum(Pauses_S[:,8] > freeze_threshold))
            # Long_Freezes_S = np.array(np.sum(Pauses_S[:,8] > long_freeze_threshold))
            # Freezes_NS = np.array(np.sum(Pauses_NS[:,8] > freeze_threshold))
            # Long_Freezes_NS = np.array(np.sum(Pauses_NS[:,8] > long_freeze_threshold))
    
        #   TR Added 250423
            midCrossings_NS = dataobject['midCrossings_NS']
            midCrossings_S = dataobject['midCrossings_S']
            boutsAngles_NS = dataobject['boutsAngles_NS']
            boutsAngles_S = dataobject['boutsAngles_S']
            boutsDist_NS = dataobject['boutsDist_NS']
            boutsDist_S = dataobject['boutsDist_S']
            Freezes_X_NS = dataobject['Freezes_X_NS']
            Freezes_Y_NS = dataobject['Freezes_Y_NS']
            Freezes_X_S = dataobject['Freezes_X_S']
            Freezes_Y_S = dataobject['Freezes_Y_S']
            
            # Make an array with all summary stats (IF VPI IS BELOW ABSOLUTE OF 0.8)
            good_fish=True
            if np.abs(VPI_NS)<VPI_Thresh:
                
                # Proportion of turns > 9.75 degrees
                angles_NS=boutsAngles_NS
                angles_S=boutsAngles_S
                propTurns_NS=np.sum(angles_NS>turnThresh)/len(angles_NS)
                propTurns_S=np.sum(angles_S>turnThresh)/len(angles_S)
                if propTurns_NS == np.nan:
                    propTurns_NS=0
                    good_fish=False
                if propTurns_S == np.nan:
                    good_fish=False
                    propTurns_S=0    
                
                # Compute mean absolute angle and distances for bouts
                # remove weird outliers from angles and distances:
                boo = np.abs(boutsAngles_NS)<270
                boutsAngles_NS=boutsAngles_NS[boo]
                boutsDist_NS=boutsDist_NS[boo]
                boo = np.abs(boutsAngles_S)<270
                boutsAngles_S=boutsAngles_S[boo]
                boutsDist_S=boutsDist_S[boo]
                # boo = midCrossings_NS < 400 and 
                # midCrossings_NS = midCrossings_NS[boo]
                # boo = midCrossings_S < 400
                # midCrossings_S = midCrossings_S[boo]
                    
                boutsAngles_NS=boutsAngles_NS[boutsAngles_NS<270]
                
                boutAbsAngleMean_NS=np.mean(np.abs(boutsAngles_NS))
                boutAbsAngleMean_S=np.mean(np.abs(boutsAngles_S))
                boutDistMean_NS=np.mean(np.abs(boutsDist_NS))
                boutDistMean_S=np.mean(np.abs(boutsDist_S))
                    
                VPI_NS_ALL.append(VPI_NS)
                VPI_S_ALL.append(VPI_S)
                BPS_NS_ALL.append(BPS_NS)
                BPS_S_ALL.append(BPS_S)
                Distance_NS_ALL.append(Distance_NS)
                Distance_S_ALL.append(Distance_S)
                Percent_Moving_NS_ALL.append(Percent_Moving_NS)
                Percent_Moving_S_ALL.append(Percent_Moving_S)
                Freezes_NS_ALL.append(Freezes_NS)
                Freezes_S_ALL.append(Freezes_S)
                Long_Freezes_NS_ALL.append(Long_Freezes_NS)
                Long_Freezes_S_ALL.append(Long_Freezes_S)
                
                # check midcrossings isn't crazy (over 1000)
                if midCrossings_NS>1000: midCrossings_NS=0
                if midCrossings_S>1000: midCrossings_S=0
                midCrossings_NS_ALL.append(midCrossings_NS)
                midCrossings_S_ALL.append(midCrossings_S)
                boutAbsAngleMean_NS_ALL.append(boutAbsAngleMean_NS)
                boutAbsAngleMean_S_ALL.append(boutAbsAngleMean_S)
                boutDistMean_NS_ALL.append(boutDistMean_NS)
                boutDistMean_S_ALL.append(boutDistMean_S)
                propTurns_NS_ALL.append(propTurns_NS)
                propTurns_S_ALL.append(propTurns_S)
                
                boutAngles_NS_ALL.append(boutsAngles_NS)
                boutAngles_S_ALL.append(boutsAngles_S)
                boutDists_NS_ALL.append(boutsDist_NS)
                boutDists_S_ALL.append(boutsDist_S)
                
                # if genes[i] == 'Scrambled':
                    # AllFish_scram.append([genes[i],float(VPI_NS),float(VPI_S),float(Freezes_NS),float(Freezes_S),float(Percent_Moving_NS),float(Percent_Moving_S),float(Distance_NS),float(Distance_S),float(midCrossings_NS),float(midCrossings_S),float(BPS_NS),float(BPS_S),float(boutDistMean_NS),float(boutDistMean_S),float(boutAbsAngleMean_NS),float(boutAbsAngleMean_S),float(propTurns_NS),float(propTurns_S),np.array(boutsDist_NS),np.array(boutsDist_S),np.array(boutsAngles_NS),np.array(boutsAngles_S)])
                # else:
                    # AllFish_cond.append([genes[i],float(VPI_NS),float(VPI_S),float(Freezes_NS),float(Freezes_S),float(Percent_Moving_NS),float(Percent_Moving_S),float(Distance_NS),float(Distance_S),float(midCrossings_NS),float(midCrossings_S),float(BPS_NS),float(BPS_S),float(boutDistMean_NS),float(boutDistMean_S),float(boutAbsAngleMean_NS),float(boutAbsAngleMean_S),float(propTurns_NS),float(propTurns_S),np.array(boutsDist_NS),np.array(boutsDist_S),np.array(boutsAngles_NS),np.array(boutsAngles_S)])
                if good_fish:
                    AllFish.append([genes[i],float(VPI_NS),float(VPI_S),float(Freezes_NS),float(Freezes_S),float(Percent_Moving_NS),float(Percent_Moving_S),float(Distance_NS),float(Distance_S),float(midCrossings_NS),float(midCrossings_S),float(BPS_NS),float(BPS_S),float(boutDistMean_NS),float(boutDistMean_S),float(boutAbsAngleMean_NS),float(boutAbsAngleMean_S),float(propTurns_NS),float(propTurns_S),np.array(boutsDist_NS),np.array(boutsDist_S),np.array(boutsAngles_NS),np.array(boutsAngles_S)])
                                
            else:
                count+=1
            # list_of_lists=[VPI_NS_ALL,VPI_S_ALL,BPS_NS_ALL,BPS_S_ALL,Distance_NS_ALL,Percent_Moving_NS_ALL,Percent_Moving_S_ALL,Freezes_NS_ALL,Freezes_S_ALL,Long_Freezes_NS_ALL,Long_Freezes_S_ALL]
            
            # for il,li in enumerate(list_of_lists):
                # list_of_lists[il]=np.array(li)
            # [VPI_NS_ALL,VPI_S_ALL,BPS_NS_ALL,BPS_S_ALL,Distance_NS_ALL,Percent_Moving_NS_ALL,Percent_Moving_S_ALL,Freezes_NS_ALL,Freezes_S_ALL,Long_Freezes_NS_ALL,Long_Freezes_S_ALL]=list_of_lists
            
            # Collect list of fish for dataframe
            # AllFish.append([genes[i],float(VPI_NS),float(VPI_S),float(BPS_NS),float(BPS_S),float(Distance_NS),float(Distance_S),float(Percent_Moving_NS),float(Percent_Moving_S),Freezes_NS,Freezes_S])
            # print('Finished '+ str(f+1) + ' of ' + str(len(npzFiles)+1))
        
        # flatten bout dists and angles
        boutDists_NS_ALL = [x for sx in boutDists_NS_ALL for x in sx]
        boutDists_S_ALL = [x for sx in boutDists_S_ALL for x in sx]
        boutAngles_NS_ALL = [x for sx in boutAngles_NS_ALL for x in sx]
        boutAngles_S_ALL = [x for sx in boutAngles_S_ALL for x in sx]
        
        count_removals.append(count)
        # Add to summary lists for plots
        VPI_NS_summary.append(VPI_NS_ALL)
        VPI_S_summary.append(VPI_S_ALL)
        
        BPS_NS_summary.append(BPS_NS_ALL)
        BPS_S_summary.append(BPS_S_ALL)
        
        Distance_NS_summary.append(Distance_NS_ALL)
        Distance_S_summary.append(Distance_S_ALL)
        
        Freezes_NS_summary.append(Freezes_NS_ALL)
        Freezes_S_summary.append(Freezes_S_ALL)
    
        Percent_Moving_NS_summary.append(Percent_Moving_NS_ALL)
        Percent_Moving_S_summary.append(Percent_Moving_S_ALL)
        
        # Long_Freezes_NS_summary.append(Long_Freezes_NS_ALL)
        # Long_Freezes_S_summary.append(Long_Freezes_S_ALL)
        
        midCrossings_NS_summary.append(midCrossings_NS_ALL)
        midCrossings_S_summary.append(midCrossings_S_ALL)
        
        propTurns_NS_summary.append(propTurns_NS_ALL)
        propTurns_S_summary.append(propTurns_S_ALL)
        
        boutAbsAngleMean_NS_summary.append(boutAbsAngleMean_NS_ALL)
        boutAbsAngleMean_S_summary.append(boutAbsAngleMean_S_ALL)
        
        boutDistMean_NS_summary.append(boutDistMean_NS_ALL)
        boutDistMean_S_summary.append(boutDistMean_S_ALL)
        
        boutDists_NS_summary.append(boutDists_NS_ALL)
        boutDists_S_summary.append(boutDists_S_ALL)
        boutAngles_NS_summary.append(boutAngles_NS_ALL)
        boutAngles_S_summary.append(boutAngles_S_ALL)
        
        print(str(genes[i])+' finished')
        # ----------------
    dfAll=pd.DataFrame(AllFish,columns=['Genotype','VPI_NS','VPI_S','Freezes_NS','Freezes_S','Percent_Moving_NS','Percent_Moving_S','Distance_NS','Distance_S', 'midCrossings_NS','midCrossings_S', 'BPS_NS','BPS_S','MeanBoutDistance_NS','MeanBoutDistance_S','MeanBoutAngle_NS','MeanBoutAngle_S','propTurns_NS','propTurns_S','boutDists_NS','boutDists_S','boutAngles_NS','boutAngles_S' ]) 
    # ZScore computations
    dfCOPY=dfAll.copy()
    dfZ=dfAll.copy()
    bouts_df=dfAll.copy()
    # extract scrambled
    numeric_cols = dfAll.select_dtypes(include=[np.number]).columns # select numeric (not labels)
    # Seperate boutAngles and boutDists
    excList=['boutAngles_NS','boutAngles_S','boutDists_NS','boutDists_S']
    keys=list(dfAll)
    for exc in excList:
        dfZ=dfZ.drop(exc,axis=1)
    excListG=excList
    excListG.append('Genotype')
    for key in keys:
        if key not in excListG:
            bouts_df.drop(key,axis=1)
    
    scram = dfAll.loc[(dfAll['Genotype']=='Scrambled')]
    # compute mean and SD
    # normalise dataframe by zScore
    for col in numeric_cols:
        scramMean=np.mean(scram[col])
        scramstd=np.std(scram[col])
        # dfSub[col]=dfAll[col]-scramMean
        dfZ[col]=(dfAll[col]-scramMean)/scramstd# zscore all other groups based on scrambled
    
    # Remove weird outliers
    boo=dfZ.Distance_NS<7
    print('Removed ' + str((np.sum(boo)-len(boo))*-1) + 'fish for distance reasons')
    dfAll = dfAll[boo]
    dfZ = dfZ[boo]
    
    # Remove weird outliers
    boo=dfAll.midCrossings_NS<1000
    print('Removed ' + str((np.sum(boo)-len(boo))*-1) + 'fish for weird midCrossings reasons')
    dfAll = dfAll[boo]
    dfZ = dfZ[boo]
    
    # IQR computations
    dfIQR=dfZ.copy()
    for col in numeric_cols:
        scramMed=np.nanmedian(scram[col])
        scramIQR=np.nanpercentile(scram[col], 75) - np.nanpercentile(scram[col], 25)
        dfIQR[col]=(dfAll[col]-scramMed)/scramIQR# IQR score all other groups based on scrambled
    # save dataframes
    if saveDataframe:
        print('saving to file...')
        dfAll.to_pickle("All_SCZ_Fish_" + date + ".pkl")
        dfZ.to_pickle("All_SCZ_Fish_Zscore_" + date + ".pkl")
        dfIQR.to_pickle("All_SCZ_Fish_IQR_" + date + ".pkl")
        bouts_df.to_pickle("All_SCZ_Fish_Bouts_" + date)
        print('saved dataframes')
# %% Stats
dfStats=dfAll.copy()
excList=['boutAngles_NS','boutAngles_S','boutDists_NS','boutDists_S']
for exc in excList:
    dfStats=dfStats.drop(exc,axis=1)
MannW_results_all_corr = SCZA.mann_whitney_all_vs_scrambled(dfStats)



#%% plot all parameters in summary violin or box plots individual figures... testing
def ViolinBoxSummaryIndFigs(dfZ,meth='box',genes=None):
    
    params=dfZ.keys()[1:]
    loops=1
    if meth=='both':
        loops=2
    for loop in range(loops):
        
        if loops==2 and loop==0:
            meth='violin'
        if loops==2 and loop==1:
            meth='box'
        for param in params:
            plt.figure()
        
            if meth=='violin':
                sns.violinplot(x='Genotype', y=param, data=dfZ,order=genes)
            elif meth=='box':
                sns.boxplot(x='Genotype', y=param, data=dfZ,order=genes)
            if param == 'propTurns_NS' or param == 'propTurns_S':
                plt.ylim(0.2,0.6)
            if param == 'MeanBoutDistance_NS' or param == 'MeanBoutDistance_S':
                plt.ylim(0,60)
            if param == 'MeanBoutAngle_NS' or param == 'MeanBoutAngle_S':
                plt.ylim(0,80)
            if param == 'midCrossings_NS' or param == 'midCrossings_S':
                plt.ylim(-2,150)
            if param == 'VPI_NS' or param == 'VPI_S':
                plt.ylim(-1.1,1.1)
            if param == 'Freezes_NS' or param == 'Freezes_S':
                plt.ylim(-0.1,40)
            if param == 'BPS_NS' or param == 'BPS_S':
                plt.ylim(-0.1,6)
            if param == 'Distance_NS' or param == 'Distance_S':
                plt.ylim(0,15000)
            if param == 'Percent_Moving_NS' or param == 'Percent_Moving_S':
                plt.ylim(0,60)
            plt.title(param)
                
    ax=plt.gca()
    return ax
# plot all parameters in summary violin plots... testing
def ViolinBoxSummarySubplots(dfZ,meth='box',genes=None):
    
    numParams=dfZ.shape[1]
    params=dfZ.keys()[1:]
    rowsNum=int(np.floor(np.sqrt(numParams)))
    colsNum=int(np.ceil(np.sqrt(numParams)))
    
    rows,cols = 0, 0
    loops=1
    if meth=='both':
        loops=2
    for loop in range(loops):
        fig, axes = plt.subplots(rowsNum,colsNum)
        if loops==2 and loop==0:
            meth='violin'
        if loops==2 and loop==1:
            meth='box'
            
        for i,param in enumerate(params):
            if np.mod(i+1,colsNum) == 0 and i>0: # if we run out of columns 
                rows+=1 # then we need to switch rows
                cols=0 # and come back to column 0
            else: # otherwise move column
                cols+=1
            if meth=='violin':
                sns.violinplot(x='Genotype', y=param, data=dfZ, ax=axes[rows,cols])
            elif meth=='box':
                sns.boxplot(x='Genotype', y=param, data=dfZ, ax=axes[rows,cols])
                
            axes[rows,cols].set_title(param)
    plt.show()

#%%
# ViolinBoxSummarSubplots(dfAll,meth='violin')
# ViolinBoxSummarySubplots(dfStats,meth='box')
numGenes=len(dfZ.groupby('Genotype'))

# ViolinBoxSummaryIndFigs(dfAll,meth='violin')
ViolinBoxSummaryIndFigs(dfStats,meth='box',genes=genes)
#%% ZScore
def heatmapBehaviour(df,MannW_results_all_corr,figname=None,vmin=-1.8,vmax=1.8,col='Spectral',sort=False): #diff=True add if trying experimental sorting
    df_GeneMean=df.groupby('Genotype').mean()
    plt.figure(figname)
    
    # custOrder=['VPI','Freezes','Percent_Moving', 'Distance','midCrossings','BPS','MeanBoutDistance','MeanBoutAngle','propTurns']
    # NONFUNCTIONAL AT PRESENT Experimental sorting
    # order=[]
    # if diff:
    #     for cust in custOrder:
    #         order.append(cust+'_Diff')
    # else:
    #     for cust in custOrder:
    #         order.append(cust+'_NS')
    #         order.append(cust+'_S')
    # index=[]
    # for i in np.arange(0,len(order)):index.append(i)
    # orderDict = dict()
    # for ind,orde in enumerate(order):
    #     orderDict[orde] = index[ind]    
    # df.sort_values(by=[key=lambda x: x.map(orderDict),axis=1)
    
    # if sort:df_GeneMean = df_GeneMean.reindex(sorted(df_GeneMean.columns), axis=1)
    df_GeneMean=df_GeneMean.drop('Scrambled')
    
    ax = sns.heatmap(df_GeneMean,vmin=vmin,vmax=vmax,cmap=col, linewidths=0.4, linecolor='grey')
    
    MannW_results_all_corr.Column=pd.Categorical(MannW_results_all_corr.Column,categories=MannW_results_all_corr.Column.unique(),ordered=True)
    df_pivot = pd.pivot_table(MannW_results_all_corr, values='pvalue_corr', index=['Genotype'], columns=['Column'])
    # df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1)

    for i in range(0,df_GeneMean.shape[0]):
        for j in range(df_GeneMean.shape[1]):
            print(str(i) + ' ' + str(j))
            value = df_pivot.iloc[i, j]
            if value < 0.001:
                # print('*** at point ' + str(j)+','+str(i))
                ax.text(j+0.5, i+0.5, "**", ha="center", va="center", color='black', fontweight='bold')
            elif value < 0.01:
                # print('** at point ' + str(j)+','+str(i))
                ax.text(j+0.5, i+0.5, "*", ha="center", va="center", color='black', fontweight='bold')
            elif value < 0.05:
                # print('* at point ' + str(j)+','+str(i))
                ax.text(j+0.5, i+0.5, "*", ha="center", va="center", color='black', fontweight='bold')
    return ax
#%% IQR (more appropriate for non-normal data)
# axZ=heatmapBehaviour(dfZ,MannW_results_all_corr,figname='ZScored',col='Spectral')
# axIQR=heatmapBehaviour(dfIQR,MannW_results_all_corr,figname='IQR',col='Spectral',vmin=-2,vmax=2)

#%% Simple difference between social and non social phases - helps adjust for baseline changes rather than changes due to the social cue. 
diff_dfZ=SCZA.compute_phase_difference(dfZ)
diff_dfIQR=SCZA.compute_phase_difference(dfIQR)
MannW_results_diff_corrZ = SCZA.mann_whitney_all_vs_scrambled(diff_dfZ)
MannW_results_diff_corrIQR = SCZA.mann_whitney_all_vs_scrambled(diff_dfIQR)
# axDiffZ=heatmapBehaviour(diff_dfZ,MannW_results_diff_corrZ,figname='ZScored_Diff')
# axDiffIQR=heatmapBehaviour(diff_dfIQR,MannW_results_diff_corrIQR,figname='IQR_Diff')
#%% append to dfZ and dfIQR for complete fingerprint
dfzz=pd.concat([dfZ,diff_dfZ],axis=1)
dfzz = dfzz.loc[:,~dfzz.columns.duplicated()].copy()
dfqq=pd.concat([dfIQR,diff_dfIQR],axis=1)
dfqq = dfqq.loc[:,~dfqq.columns.duplicated()].copy()
MannW_results_all_corr_fullZ=MannW_results_all_corr.append(MannW_results_diff_corrZ)
MannW_results_all_corr_fullIQR=pd.concat([MannW_results_all_corr,MannW_results_diff_corrIQR],axis=0)

# plt.close('ZScored')
# plt.close('IQR')
axZ=heatmapBehaviour(dfzz,MannW_results_all_corr_fullZ,figname='ZScored_fdr_bh_corrected')
axIQR=heatmapBehaviour(dfqq,MannW_results_all_corr_fullZ,figname='IQR_fdr_bh_corrected')

# diffZax=heatmapBehaviour(diff_dfZ,MannW_results_diff_corrZ,figname='Zscore Phase Diff')
# diffIQRax=heatmapBehaviour(diff_dfIQR,MannW_results_diff_corrIQR,figname='IQR Phase Diff')
#%%
# cmap.set_under('darkred')
# fig, ax = plt.subplots()
# cax = ax.imshow(df_pivot, cmap=cmap, vmin=0, vmax=0.1)

# plt.figure()
# ax1 = sns.heatmap(df_pivot,vmin=0,vmax=0.1,cmap='Spectral', linewidths=0.4, linecolor='grey')


# for _, spine in ax.spines.items():
#     spine.set_visible(True)

# for item in ax.get_yticklabels():
#     item.set_rotation(0)

# for item in ax.get_xticklabels():
#     item.set_rotation(45)
    
# Freezes location; important where they are

# Find a way to cover the baseline (NS) changes in the social side (i.e. if fish are swimming less in the NS, then you would expect them to be less in S)

# Per bout metrics! Comparison plots and distance vs angle - short bouts are sign of stress


#%%    Data collected, now some helper functions to run stats and plots
# 
#------------------------
# Summary plots and stats

    
    


#%% Run stats and plots for VPI
# plotType='nothing'
plotType='swarm'
# plotType='strip'
att='_dot'
mainPlot='dot'
subplotNum=(2,3,1)
errMode=0.95
# subplotNum=(0,0,0)
variable='VPI'
NS_data=VPI_NS_summary
S_data=VPI_S_summary
ylim=None
df,df_p=SCZA.plotOverallComparison(subplotNum,variable,conditionNames,NS_data,S_data,errMode=errMode,mainPlot=mainPlot,plotType=plotType,ylim=ylim)
df_cond=pd.DataFrame({'Genotype' : conditionNames})
df_p_VPI=pd.concat([df_cond,df_p],axis=1)
df_p_All=pd.concat([df_cond,df_p],axis=1)
# Run stats and plots for BPS
subplotNum=(2,3,2)
variable='BPS'
NS_data=BPS_NS_summary
S_data=BPS_S_summary
ylim=(0,5)
df,df_p=SCZA.plotOverallComparison(subplotNum,variable,conditionNames,NS_data,S_data,errMode=errMode,mainPlot=mainPlot,plotType=plotType,ylim=ylim)
df_p_BPS=pd.concat([df_cond,df_p],axis=1)
df_p_All=pd.concat([df_p_All,df_p],axis=1)
# Run stats and plots for Distance
subplotNum=(2,3,3)
variable='DistanceTraveled'
NS_data=Distance_NS_summary
S_data=Distance_S_summary
ylim=(0,15000)
df,df_p=SCZA.plotOverallComparison(subplotNum,variable,conditionNames,NS_data,S_data,errMode=errMode,mainPlot=mainPlot,plotType=plotType,ylim=ylim)
df_p_Dist=pd.concat([df_cond,df_p],axis=1)
df_p_All=pd.concat([df_p_All,df_p],axis=1)
# Run stats and plots for Freezes
subplotNum=(2,3,4)
variable='Freezes'
NS_data=Freezes_NS_summary
S_data=Freezes_S_summary
ylim=(0,20)
df,df_p=SCZA.plotOverallComparison(subplotNum,variable,conditionNames,NS_data,S_data,errMode=errMode,mainPlot=mainPlot,plotType=plotType,ylim=ylim)
df_p_Freezes=pd.concat([df_cond,df_p],axis=1)
df_p_All=pd.concat([df_p_All,df_p],axis=1)
# Run stats and plots for Percent Moving
# subplotNum=(2,3,5)
# variable='PercentTimeMoving'
# NS_data=Percent_Moving_NS_summary
# S_data=Percent_Moving_S_summary
# ylim=None
# df,df_p=SCZA.plotOverallComparison(subplotNum,variable,conditionNames,NS_data,S_data,errMode=errMode,mainPlot=mainPlot,plotType=plotType,ylim=ylim)
# df_p_PercMov=pd.concat([df_cond,df_p],axis=1)
# df_p_All=pd.concat([df_p_All,df_p],axis=1)
# Run stats and plots for midcrossings
subplotNum=(2,3,5)
variable='midCrossings'
NS_data=Percent_Moving_NS_summary
S_data=Percent_Moving_S_summary
ylim=None
df,df_p=SCZA.plotOverallComparison(subplotNum,variable,conditionNames,NS_data,S_data,errMode=errMode,mainPlot=mainPlot,plotType=plotType,ylim=ylim)
df_p_PercMov=pd.concat([df_cond,df_p],axis=1)
df_p_All=pd.concat([df_p_All,df_p],axis=1)
# Difference in VPI
subplotNum=(2,3,6)
variable='VPI Difference'
NS_data=VPI_NS_summary
S_data=VPI_S_summary

dfDiff,p_Diff=SCZA.plot_VPI_diff(subplotNum,variable,conditionNames,NS_data,S_data,ylim=None,mainPlot='dot',plotType='',att='',size=2,baralpha=0.8,wid=0.2,errMode='se',conf_meth='',alpha=0.4)

# new figures for bouts  & midCrossings
att='_Bouts_dot'
plotType='nothing'
mainPlot='dot'
errMode=0.95
# Run stats and plots for midcrossings
subplotNum=(2,2,1)
variable='midCrossings'
NS_data=midCrossings_NS_summary
S_data=midCrossings_S_summary
ylim=None
df,df_p=SCZA.plotOverallComparison(subplotNum,variable,conditionNames,NS_data,S_data,errMode=errMode,mainPlot=mainPlot,plotType=plotType,ylim=ylim)
df_p_midCrossings=pd.concat([df_cond,df_p],axis=1)
df_p_All=pd.concat([df_p_All,df_p],axis=1)

subplotNum=(2,2,2)
variable='MeanBoutAngles'
NS_data=boutAbsAngleMean_NS_summary
S_data=boutAbsAngleMean_S_summary
ylim=None
df,df_p=SCZA.plotOverallComparison(subplotNum,variable,conditionNames,NS_data,S_data,errMode=errMode,mainPlot=mainPlot,plotType=plotType,ylim=ylim)
df_p_meanBoutAngle=pd.concat([df_cond,df_p],axis=1)
df_p_All=pd.concat([df_p_All,df_p],axis=1)

subplotNum=(2,2,3)
variable='MeanBoutDists'
NS_data=boutDistMean_NS_summary
S_data=boutDistMean_S_summary
ylim=None
df,df_p=SCZA.plotOverallComparison(subplotNum,variable,conditionNames,NS_data,S_data,errMode=errMode,mainPlot=mainPlot,plotType=plotType,ylim=ylim)
df_p_meanBoutAngle=pd.concat([df_cond,df_p],axis=1)
df_p_All=pd.concat([df_p_All,df_p],axis=1)

subplotNum=(2,2,4)
variable='propTurns'
NS_data=propTurns_NS_summary
S_data=propTurns_S_summary
ylim=None
df,df_p=SCZA.plotOverallComparison(subplotNum,variable,conditionNames,NS_data,S_data,errMode=errMode,mainPlot=mainPlot,plotType=plotType,ylim=ylim)
df_p_meanBoutAngle=pd.concat([df_cond,df_p],axis=1)
df_p_All=pd.concat([df_p_All,df_p],axis=1)


# df_p_All=pd.concat(df_p_All,df_p)
# tukey = pairwise_tukeyhsd(endog=df['score'],groups=df['group'],alpha=0.05)

# Run stats and plots for Distance per bout
# Run stats and plots for Angle per bout
# Make angle vs distance plots for each genotype


#%%
print('FIN')
#FIN

#%%
       
    ####### WIP ##########
    # series_list_NS = []
    # series_list_S = []
    # series_list_ALL = []
    # series_list = []
    # geneS=[]
    # xxS=[]
    
    # gene=[]
    # xx=[]
    
    # palette=sns.color_palette("hls", len(conditionNames))
    # for i, name in enumerate(conditionNames):
    #     s = pd.Series(NS_data[i], name=variable)
    #     for j in range(len(NS_data[i])):
    #         gene.append(name)
    #         xx.append("NS: " + name)
    #     series_list.append(s)
        
    
    # for i, name in enumerate(conditionNames):
    #     condition.append(i)
    #     s = pd.Series(S_data[i], name="S: " + name)
    #     for j in range(len(S_data[i])):
    #         gene.append(name)
    #         xx.append("S: " + name)
    #     series_list.append(s)
        
    # df = pd.concat(series_list, axis=1)
        
    # sns.barplot(x=xx,y=df, orient="v", saturation=0.2, color=(0.75,0.75,0.75,0.8), ci=95, capsize=0.05, errwidth=2,hue=geneS)
    # if plotType=='strip':
    #     sns.stripplot(data=df, orient="v", size=2, jitter=True, dodge=True, edgecolor="white", hue=gene,palette=palette)
    # elif plotType=='swarm':
    #     sns.swarmplot(data=df, orient="v", size=1, edgecolor="white",color='gray')
    # plt.xticks(rotation=45)
    
# Make a big dataframe of everything???

# loop through each fish and create new entry for it

# columnNames=['Gene','NS_VPI','NS_BPS','NS_DistanceTravelled','NS_PercentTimeMoving','NS_Freezes','NS_LongFreezes','S_VPI','S_BPS','S_DistanceTravelled','S_PercentTimeMoving','S_Freezes','S_LongFreezes']
# gene_list=[]
# for i, name in enumerate(conditionNames):
#     gene_list.append(name)
#     # [NS_VPI,NS_BPS,NS_DistanceTravelled,NS_PercentTimeMoving,NS_Freezes,NS_LongFreezes,S_VPI,S_BPS,S_DistnaceTravelled,S_PercentTimeMoving,S_Freezes,S_LongFreezes]
#     varlist=[VPI_NS_summary[i],BPS_NS_summary[i],Distance_NS_summary[i],Percent_Moving_NS_summary[i],Freezes_NS_summary[i],Long_Freezes_NS_summary[i],VPI_S_summary[i],BPS_S_summary[i],Distance_S_summary[i],Percent_Moving_S_summary[i],Freezes_S_summary[i],Long_Freezes_S_summary[i]]
#     for j,colName in columnNames:
#         NS_VPI_List.append(varlist[j])
