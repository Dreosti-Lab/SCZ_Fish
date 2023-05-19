# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 10:50:31 2023

@author: Tom Ryan, UCL (Dreosti Group)

Sleep analysis scripts after running through Rihel-lab's Frame by Frame pipeline in R
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

genes=['Scrambled','xpo7']
cols=['black','magenta']
plotall=True
path='S:/WIBR_Dreosti_Lab/Tom/Crispr_Project/Behavior/Sleep_experiments/TomSleep/zzztc_220919_11_gria3xpo7_epo10.csv'

zzzData=pd.read_csv(path)
zDataArray=np.array(zzzData)
colNames=zzzData.keys() # grab names

# grab gene columns specified
toKeep=[]
for colNum,name in enumerate(colNames):
    d=zzzData[name]
    # print(d[0])
    for geneCount,gene in enumerate(genes):
        if d[0]==gene:
            toKeep.append(True)
    if len(toKeep)==colNum or len(toKeep)==0: toKeep.append(False)
    
zDataArray=zDataArray.T[toKeep]
zzDataArray=zDataArray[:,1:]
geneList=zDataArray[:,0]
zzDataArray=zzDataArray.astype(float)
# print(geneList)


def grabTraces(zzDataArray,genes,method='std'):
    
    allTraces=[]
    meanTraces=[]
    varis=[]
    for gene in genes:
        traces=zzDataArray[geneList==gene]
        allTraces.append(traces)
        meanTraces.append(np.mean(traces,axis=0))
        std=np.std(traces,axis=0)
        if method=='std':
            varis.append(std)
        elif method=='sem':
            varis.append(std/np.sqrt(traces.shape[0]))
    return allTraces,meanTraces,varis

allTraces,meanTraces,varis=grabTraces(zzDataArray,genes,method='sem')
meanID=len(genes)
fig, axs = plt.subplots(meanID+1,2, sharex=True, sharey=True)

for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        for spine in ['top', 'right']:        
            axs[i,j].spines[spine].set_visible(False)
            # axs[i,j].vlines(transitions)
            axs[i,j].grid()
# axs[0, 0].plot(x, y)
# axs[0, 0].set_title('Axis [0, 0]')
# axs[0, 1].plot(x, y, 'tab:orange')
# axs[0, 1].set_title('Axis [0, 1]')
# axs[1, 0].plot(x, -y, 'tab:green')
# axs[1, 0].set_title('Axis [1, 0]')
# axs[1, 1].plot(x, -y, 'tab:red')
# axs[1, 1].set_title('Axis [1, 1]')
for geneCount,gene in enumerate(genes):
    if plotall:
        for traces in allTraces[geneCount]:
            y=traces
            axs[geneCount,0].plot(y,color=cols[geneCount],linewidth=.5,alpha=0.04)
            axs[meanID,0].plot(y,color=cols[geneCount],linewidth=.5,alpha=0.04)
            
    # plot means and variances
    y=meanTraces[geneCount]
    x=range(len(y))
    error_pos=y+varis[geneCount]
    error_neg=y-varis[geneCount]
    
    axs[geneCount,1].plot(x,y,linewidth=1.5, color=cols[geneCount],label=gene,alpha=0.8)
    axs[geneCount,1].plot(x,error_pos,linewidth=1.5, color=cols[geneCount],label=gene,alpha=0.8)
    axs[geneCount,1].plot(x,error_neg,linewidth=1.5, color=cols[geneCount],label=gene,alpha=0.8)
    axs[geneCount,1].fill_between(x,error_neg,error_pos,color=cols[geneCount],alpha=0.4)
    
    axs[meanID,1].plot(x,y,linewidth=1.5, color=cols[geneCount],label=gene,alpha=0.8)
    axs[meanID,1].plot(x,error_pos,linewidth=1.5, color=cols[geneCount],label=gene,alpha=0.8)
    axs[meanID,1].plot(x,error_neg,linewidth=1.5, color=cols[geneCount],label=gene,alpha=0.8)
    axs[meanID,1].fill_between(x,error_neg,error_pos,color=cols[geneCount],alpha=0.4)
    
    
plt.legend(loc='upper right')