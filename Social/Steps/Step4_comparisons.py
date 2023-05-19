# -*- coding: utf-8 -*-
"""
Compare summaries of analyzed social preference experiments

@author: Tom Ryan, UCL (Dreosti-Group) 
"""
# -----------------------------------------------------------------------------
lib_path = r'S:\WIBR_Dreosti_Lab\Tom\Github\SCZ_Model_Fish\libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)

# -----------------------------------------------------------------------------
# Set "Base Path" for this analysis session
base_path = r'S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/AnalysisRounds/Analysis_TestNEW'
# -----------------------------------------------------------------------------
#%%
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

genes=[]
genes.append(r'Scrambled')
# genes.append(r'nr3c2')
# genes.append(r'hcn4')
# genes.append(r'akap11')
# genes.append(r'gria3')
genes.append(r'sp4')
# genes.append(r'xpo7')
genes.append(r'trio')
# genes.append(r'cacna1g')
conditionNames=genes
analysisFolders=[]
# Set analysis folder and label for experiment/condition A
for gene in genes:
    analysisFolders.append(base_path + r'/' + gene)

VPI_Thresh=0.8
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


#%% Go through each condition (analysis folder)
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

        # Make an array with all summary stats (IF VPI IS BELOW ABSOLUTE OF 0.8)
        if np.abs(VPI_NS)<VPI_Thresh:
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
            if genes[i] == 'Scrambled':
                AllFish_scram.append([genes[i],float(VPI_NS),float(VPI_S),float(BPS_NS),float(BPS_S),float(Distance_NS),float(Distance_S),float(Percent_Moving_NS),float(Percent_Moving_S),float(Freezes_NS),float(Freezes_S)])
            else:
                AllFish_cond.append([genes[i],float(VPI_NS),float(VPI_S),float(BPS_NS),float(BPS_S),float(Distance_NS),float(Distance_S),float(Percent_Moving_NS),float(Percent_Moving_S),float(Freezes_NS),float(Freezes_S)])
            AllFish.append([genes[i],float(VPI_NS),float(VPI_S),float(BPS_NS),float(BPS_S),float(Distance_NS),float(Distance_S),float(Percent_Moving_NS),float(Percent_Moving_S),float(Freezes_NS),float(Freezes_S)])
            
        else:
            count+=1
        # list_of_lists=[VPI_NS_ALL,VPI_S_ALL,BPS_NS_ALL,BPS_S_ALL,Distance_NS_ALL,Percent_Moving_NS_ALL,Percent_Moving_S_ALL,Freezes_NS_ALL,Freezes_S_ALL,Long_Freezes_NS_ALL,Long_Freezes_S_ALL]
        
        # for il,li in enumerate(list_of_lists):
            # list_of_lists[il]=np.array(li)
        # [VPI_NS_ALL,VPI_S_ALL,BPS_NS_ALL,BPS_S_ALL,Distance_NS_ALL,Percent_Moving_NS_ALL,Percent_Moving_S_ALL,Freezes_NS_ALL,Freezes_S_ALL,Long_Freezes_NS_ALL,Long_Freezes_S_ALL]=list_of_lists
        
        # Collect list of fish for dataframe
        # AllFish.append([genes[i],float(VPI_NS),float(VPI_S),float(BPS_NS),float(BPS_S),float(Distance_NS),float(Distance_S),float(Percent_Moving_NS),float(Percent_Moving_S),Freezes_NS,Freezes_S])
        # print('Finished '+ str(f+1) + ' of ' + str(len(npzFiles)+1))
    
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
    
    Long_Freezes_NS_summary.append(Long_Freezes_NS_ALL)
    Long_Freezes_S_summary.append(Long_Freezes_S_ALL)
    print(str(genes[i])+' finished')
    # ----------------
    
dfAll=pd.DataFrame(AllFish,columns=['Genotype','VPI_NS','VPI_S','BPS_NS','BPS_S','Distance_NS','Distance_S','Percent_Moving_NS','Percent_Moving_S','Freezes_NS','Freezes_S'])
#%% 
# IQR computations
dfZ=dfAll.copy()
dfSub=dfAll.copy()
# extract scrambled
numeric_cols = dfAll.select_dtypes(include=[np.number]).columns # select numeric (not labels)
scram = dfAll.loc[(dfAll['Genotype']=='Scrambled')]
# compute mean and SD
scramMean=scram.mean()
scramstd=scram.std()
# normalise dataframe
for col in numeric_cols:
    dfSub[col]=dfAll[col]-scramMean[col]
    dfZ[col]=(dfAll[col]-scramMean[col])/scramstd[col]# normalise all other groups

# Remove weird outliers
dfAll = dfAll[dfZ.Distance_NS<7]
dfZ = dfZ[dfZ.Distance_NS<7]
saveDataframe=True
if saveDataframe:
    dfAll.to_pickle("All_SCZ_Fish_testNew.pkl")

MannW_results_all_corr = SCZA.mann_whitney_all_vs_scrambled(dfAll)

#%% plot summary violin plots... testing
numParams=dfZ.shape[1]
params=dfZ.keys()[1:]
numGenes=len(dfZ.groupby('Genotype'))
rowsNum=int(np.floor(np.sqrt(numParams)))
colsNum=int(np.ceil(np.sqrt(numParams)))
fig, axes = plt.subplots(rowsNum,colsNum)
rows,cols = 0, 0
for i,param in enumerate(params):
    if np.mod(i,colsNum) == 0 and i>0: # if we run out of columns 
        rows+=1 # then we need to switch rows
        cols=0 # and come back to column 0
    sns.violinplot(x='Genotype', y=param, data=dfZ, ax=axes[rows,cols])
    axes[rows,cols].set_title(param)
    cols+=1 # move to next column
plt.show()

#%%
# plot in box and whisker
fig, axes = plt.subplots(rowsNum,colsNum)
rows,cols = 0, 0
for i,param in enumerate(params):
    if np.mod(i,colsNum) == 0 and i>0: # if we run out of columns 
        rows+=1 # then we need to switch rows
        cols=0 # and come back to column 0
    sns.boxplot(x='Genotype', y=param, data=dfZ, ax=axes[rows,cols])
    axes[rows,cols].set_title(param)
    cols+=1 # move to next column
plt.show()

#%% plot all behaviours as a heatmap, genes against behaviours
dfZ_GeneMean=dfZ.groupby('Genotype').mean()
plt.figure()
dfZ_GeneMean['VPI_S']*=1.5
# dfZ_GeneMean['Freezes_S']*=2
# dfZ_GeneMean['Freezes_NS']*=2
dfZ_GeneMean = dfZ_GeneMean.reindex(sorted(dfZ_GeneMean.columns), axis=1)
dfZ_GeneMean_exScram=dfZ_GeneMean.drop('Scrambled')
ax = sns.heatmap(dfZ_GeneMean_exScram,vmin=-2,vmax=2,cmap='Spectral', linewidths=0.4, linecolor='grey')

# fig, ax = plt.subplots()
# cmap = plt.cm.get_cmap('Spectral', 256)
# cax = ax.imshow(dfZ_GeneMean, cmap=cmap, vmin=0, vmax=0.1)

df_pivot = pd.pivot_table(MannW_results_all_corr, values='pvalue', index=['Genotype'], columns=['Column'])
df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1)

for i in range(dfZ_GeneMean_exScram.shape[0]):
    for j in range(dfZ_GeneMean_exScram.shape[1]):
        value = df_pivot.iloc[i, j]
        if value < 0.001:
            ax.text(j+0.5, i+0.5, "***", ha="center", va="center", color='black', fontweight='bold')
        elif value < 0.01:
            ax.text(j+0.5, i+0.5, "**", ha="center", va="center", color='black', fontweight='bold')
        elif value < 0.05:
            ax.text(j+0.5, i+0.5, "*", ha="center", va="center", color='black', fontweight='bold')
                


diff_df=SCZA.compute_phase_difference(dfZ)
MannW_results_diff_corr = SCZA.mann_whitney_all_vs_scrambled(diff_df)
dfZ_GeneMean=diff_df.groupby('Genotype').mean()
plt.figure()
dfZ_GeneMean['VPI_Diff']*=1.5
# dfZ_GeneMean['Freezes_S']*=2
# dfZ_GeneMean['Freezes_NS']*=2
dfZ_GeneMean = dfZ_GeneMean.reindex(sorted(dfZ_GeneMean.columns), axis=1)
dfZ_GeneMean_exScram=dfZ_GeneMean.drop('Scrambled')
ax = sns.heatmap(dfZ_GeneMean_exScram,vmin=-2,vmax=2,cmap='Spectral', linewidths=0.4, linecolor='grey')

# fig, ax = plt.subplots()
# cmap = plt.cm.get_cmap('Spectral', 256)
# cax = ax.imshow(dfZ_GeneMean, cmap=cmap, vmin=0, vmax=0.1)

df_pivot = pd.pivot_table(MannW_results_diff_corr, values='pvalue_corr', index=['Genotype'], columns=['Column'])
df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1)

for i in range(dfZ_GeneMean_exScram.shape[0]):
    for j in range(dfZ_GeneMean_exScram.shape[1]):
        value = df_pivot.iloc[i, j]
        if value < 0.001:
            ax.text(j+0.5, i+0.5, "***", ha="center", va="center", color='black', fontweight='bold')
        elif value < 0.01:
            ax.text(j+0.5, i+0.5, "**", ha="center", va="center", color='black', fontweight='bold')
        elif value < 0.05:
            ax.text(j+0.5, i+0.5, "*", ha="center", va="center", color='black', fontweight='bold')
                



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
# Per bout metrics! - short bouts are sign of stress

#%%    Data collected, now some helper functions to run stats and plots
# 
#------------------------
# Summary plots and stats

    
    


#%% Run stats and plots for VPI
# plotType='swarm'
# plotType='strip'
att='_dot'
plotType='nothing'
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
subplotNum=(2,3,5)
variable='PercentTimeMoving'
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
