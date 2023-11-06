# -*- coding: utf-8 -*-
"""
Created on Wed May 07 19:13:12 2014

@author: Elena
#"""
## -----------------------------------------------------------------------------
## Detect Platform
#import platform
#if(platform.system() == 'Linux'):
#    # Set "Repo Library Path" - Social Zebrafish Repo
#    lib_path = r'/home/kampff/Repos/Dreosti-Lab/Social_Zebrafish/libs'
#else:
#    # Set "Repo Library Path" - Social Zebrafish Repo
#    lib_path = r'C:/Repos/Dreosti-Lab/Social_Zebrafish/libs'

lib_path = r'C:\Users\thoma\OneDrive\Documents\GitHub\Arena_Zebrafish\libs'
# Set Library Paths
import sys
sys.path.append(lib_path)
# -----------------------------------------------------------------------------
# Import custom libraries
import SCZ_math as SCZM
import SCZ_utilities as SCZU

# Import useful libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multitest import fdrcorrection

#Finish later: extract each loom bout's movie...
#def extractMoviesFromStim(aviFile,startTime=15,interval=2,duration=1,numFrames=240,frameRate=120):
#    startFrame=startTime*60*120
#    intervalFrames=interval*60*frameRate
#    durationFrames=duration*frameRate
#    firstLoomStart=startFrame+intervalFrames-durationFrames
#    loomStarts=[]
#    loomEnds=[]
#    loomStarts.append(firstLoomStart)
#    loomEnds.append(firstLoomStart+numFrames-1)
#    
#    vid=cv2.VideoCapture(aviFile)
#    
#    for i,start in loomStarts:
#        end=loomEnds[i]

#TR 06/04/23
# Compute number of midline crossings (same boundary as VPI)
def computeMidCrossings(xPositions, yPositions, testROI, stimROI, FPS=120, longhand=True):
    
    # Find thresholds of Y from the Test ROI in order to define the Visible area
    boundary = testROI[1]+(testROI[3]/2) 
    
    # check if social stim was top or bottom, and whether fish began trial in social or non social side
    socialTop = stimROI[1] < boundary
    top=False
    if yPositions[0]<boundary:
        top=True
        if socialTop:
            beganSocial=True
        else:
            beganSocial=False
    elif socialTop:
        beganSocial=False
    else:
        beganSocial=True
    
    # subtract boundary from yPositions 
    if longhand:
    # Long and safe way
        crossingCount=0
        buffer=10
        bufferTopCount=0    
        bufferBotCount=0
        for yi,y in enumerate(yPositions):
            if y>boundary and top: # if you're in the bottom and were in the top...
                bufferTopCount=0    
                bufferBotCount+=1
                if bufferBotCount>buffer:
                    # print('Gone from top to bottom at frame ' + str(yi))
                    crossingCount+=1
                    top=False
                    bufferBotCount=0
            elif y<boundary and top==False: # if you're in the top and were in the bottom...
                bufferBotCount=0
                bufferTopCount+=1
                if bufferTopCount>buffer:
                    # print('Gone from bottom to top at frame ' + str(yi))
                    crossingCount+=1
                    top=True
                    bufferTopCount=0
                    
    # Fast and slightly scary way
    else:
        yPositions_zeroed=yPositions-boundary
        countCrossings=(np.diff(np.sign(yPositions_zeroed)) != 0).sum()
    
    return countCrossings,beganSocial

def checkStatsDiffMW(df,variable,scram,conditionNames):
    pS=[]
    for i, name in enumerate(conditionNames):
        
        # Social Diff Stat different from scram?    
        group1=df[variable + ' ' + name]
        group1=group1[~np.isnan(group1)]
        group2=scram
        group2=group2[~np.isnan(group2)]
        _,p=stats.mannwhitneyu(group1, group2, alternative='two-sided')
        pS.append(p)

    return pS,conditionNames

def checkStatsDiff(df,variable,scram,conditionNames):
    dataf_val,dataf_nam=[],[]
    for i, name in enumerate(conditionNames):
        # Social Diff Stat different from scram?  
        # For some stupid reason the tukey test doesn't work on the same dataframe so we need to reshape
        group=df[variable + ' ' + name]
        group=group[~np.isnan(group)]
        dataf_val.append(group)
        dataf_nam.append(np.repeat(name,len(group)))
        
    dataf_val = [s for slist in dataf_val for s in slist]
    dataf_nam = [s for slist in dataf_nam for s in slist]
    dftemp={'val' : dataf_val,
            'name' : dataf_nam}
    
    tukey = pairwise_tukeyhsd(endog=dftemp['val'],groups=dftemp['name'],alpha=0.05)

    return tukey

# Now we can do the plots and stats on each metric
def checkStats1(df,scram_NS,scram_S,conditionNames):
    p_fromNS,p_fromScramNS,p_fromScramS=[],[],[]
    for i, name in enumerate(conditionNames):
        
        # Social phase Stat different from NonSocial phase?  
        group1=df['NS: ' + name]
        group1=group1[~np.isnan(group1)]
        group2=df['S: ' + name]
        group2=group2[~np.isnan(group2)]
        _,p=stats.mannwhitneyu(group1, group2, alternative='two-sided')
        p_fromNS.append(p)
        
        # NonSocial phase Stat different from NS scram?    
        group1=df['NS: ' + name]
        group1=group1[~np.isnan(group1)]
        group2=scram_NS
        group2=group2[~np.isnan(group2)]
        _,p=stats.mannwhitneyu(group1, group2, alternative='two-sided')
        p_fromScramNS.append(p)
        
        # Social phase Stat different from S scram?    
        group1=df['S: ' + name]
        group1=group1[~np.isnan(group1)]
        group2=scram_S
        group2=group2[~np.isnan(group2)]
        _,p=stats.mannwhitneyu(group1, group2, alternative='two-sided')
        p_fromScramS.append(p)

    return p_fromNS,p_fromScramNS,p_fromScramS,conditionNames

# currently broken#
def plotOverallComparison_subplots(variable,conditionNames,NS_data,S_data,ylim=None,mainPlot='dot',plotType='swarm',fignames=['All','NS','S'],att='',size=2,baralpha=0.8,wid=0.2,errMode='se',conf_meth='',alpha=0.4):
    
    scramNS=NS_data[0]
    scramS=S_data[0]
    palette=sns.color_palette("deep", 15)
    colorsS=[[(0.75,0.75,0.75)],palette[:len(conditionNames[1:])],[(0.75,0.75,0.75)],palette[:len(conditionNames)-1]]
    colors = [i for s in colorsS for i in s]
    # colors=['gray','blue','r','g','m','y','c','darkred','gray','blue','r','g','m','y','c','darkred']
    
    series_list = []
    series_list_NS = []
    series_list_S = []
    
    for i, name in enumerate(conditionNames):
        s = pd.Series(NS_data[i], name="NS: " + name)
        series_list.append(s)
        series_list_NS.append(s)
    for i, name in enumerate(conditionNames):
        s = pd.Series(S_data[i], name="S: " + name)
        series_list.append(s)
        series_list_S.append(s)
        
    df = pd.concat(series_list, axis=1)
    df_NS = pd.concat(series_list_NS, axis=1)
    df_S = pd.concat(series_list_S, axis=1)
    dfList=[df,df_NS,df_S]
    
    # run stats and create dataframe to store pvalues
    # metrics=['VPI','BPS','DistanceTravelled','Freezes','PercentTimeMoving']
    # statColumns=[]
    # statColumns.append('S_from_NS_' + metric)
    # statColumns.append('NS_from_Scram_NS_'+metric)
    # statColumns.append('S_from_Scram_S_'+metric)
    p_fromNS,p_fromScramNS,p_fromScramS,p_fromNS_corr,p_fromScramNS_corr,p_fromScramS_corr=checkStats(df,scramNS,scramS,conditionNames)
    d_p={'S_from_NS_' + variable : p_fromNS_corr,
         'NS_from_Scram_NS_' + variable : p_fromScramNS_corr,
         'S_from_Scram_S_' + variable: p_fromScramS_corr}
    df_p=pd.DataFrame(d_p,columns=list(d_p.keys()))

    for figId,dfP in enumerate(dfList):
        figname=fignames[figId]+att
        plt.figure(figname,constrained_layout=True)
        plt.suptitle(figname)
        # plt.tight_layout()
        plt.subplot(subplotNum[0],subplotNum[1],subplotNum[2])
        plt.figure(figname,constrained_layout=True)
        plt.title(variable)
        if mainPlot=='bar':
            # sns.barplot(data=dfP, orient="v", saturation=0.2, color=(0.75,0.75,0.75,baralpha), ci=95, capsize=0.05, errwidth=2)
            ax=sns.barplot(data=dfP, orient="v", saturation=0.3, ci=95, capsize=0.05, errwidth=2,alpha=0.8)
            for bar, color in zip(ax.containers[0], colors):
                bar.set_color(color)
        elif mainPlot=='violin':
            ax=sns.violinplot(data=dfP, orient="v", saturation=0.3, ci=95, capsize=0.05, errwidth=2,alpha=0.8)
            
        elif mainPlot=='dot':
            # find x of data
            colNames=dfP.columns.tolist()
            x=np.linspace(0,len(colNames)-1,len(colNames))
            # find y postiiton of mean and std err + and -
            y_mean=np.nanmean(np.asarray(dfP,dtype=np.float64),axis=0)
            y_std=np.nanstd(np.asarray(dfP,dtype=np.float64),axis=0)
            
            # work out n for each column
            leng=dfP.shape[0]
            n=leng-(dfP.isna().sum())
            y_ste=y_std/np.sqrt(n)
            
            # draw points - circle for mean, line for err
            for idcol,color in enumerate(colors):
                xd=x[idcol]
                yd=y_mean[idcol]
                if errMode=='std':
                    ye=y_std[idcol]
                elif errMode=='se':
                    ye=y_ste[idcol]
                elif type(errMode)==float:
                    dat=np.asarray(dfP,dtype=np.float64)[:,idcol]
                    dat = dat[~np.isnan(dat)]
                    if conf_meth=='bootstrap':
                        # compute 95% CI by 1k bootstrap...
                        values = [np.random.choice(dat,size=len(x),replace=True).mean() for i in range(1000)] 
                        ye=np.percentile(values,[100*(1-errMode)/2,100*(1-(1-errMode)/2)])[1]-yd
                    else:
                        # or by t-distribution
                        ye=stats.t.interval(confidence=errMode, df=n[idcol], loc=yd, scale=y_ste[idcol])[1]-yd
                
                colName=colNames[idcol]
                plt.scatter(xd,yd,color=color,label=colName)
                # draw line between (x,y-err):(x,y+err)
                plt.plot((xd,xd),(yd-ye,yd+ye),color=color)
                plt.plot((xd-wid,xd+wid),(yd-ye,yd-ye),color=color)
                plt.plot((xd-wid,xd+wid),(yd+ye,yd+ye),color=color)
                
            plt.xticks(x, colNames)
        
        if plotType=='strip':
            plt.subplot(subplotNum[0],subplotNum[1],subplotNum[2])
            sns.stripplot(data=dfP, orient="v", size=2, jitter=True, dodge=True, edgecolor="white",alpha=alpha,color='gray')
            plt.xticks(rotation=45)
        
        elif plotType=='swarm':
            plt.subplot(subplotNum[0],subplotNum[1],subplotNum[2])
            sns.swarmplot(data=dfP, orient="v", size=1,alpha=alpha, color='gray')
            plt.xticks(rotation=45)
        
        if ylim is not None:
            plt.ylim(ylim)
        return df, df_p

def plotOverallComparison(variable, conditionNames, NS_data, S_data, ylim=None, mainPlot='dot', plotType='swarm', fignames=['All', 'NS', 'S'], att='', size=2, baralpha=0.8, wid=0.2, errMode='se', conf_meth='', alpha=0.4):

    scramNS = NS_data[0]
    scramS = S_data[0]
    palette = sns.color_palette("deep", 15)
    colorsS = [[(0.75, 0.75, 0.75)], palette[:len(conditionNames[1:])], [(0.75, 0.75, 0.75)], palette[:len(conditionNames) - 1]]
    colors = [i for s in colorsS for i in s]

    series_list = []
    series_list_NS = []
    series_list_S = []

    for i, name in enumerate(conditionNames):
        s = pd.Series(NS_data[i], name="NS: " + name)
        series_list.append(s)
        series_list_NS.append(s)
    for i, name in enumerate(conditionNames):
        s = pd.Series(S_data[i], name="S: " + name)
        series_list.append(s)
        series_list_S.append(s)

    df = pd.concat(series_list, axis=1)
    df_NS = pd.concat(series_list_NS, axis=1)
    df_S = pd.concat(series_list_S, axis=1)
    dfList = [df, df_NS, df_S]

    # Run stats and create dataframe to store p-values
    p_fromNS, p_fromScramNS, p_fromScramS, p_fromNS_corr, p_fromScramNS_corr, p_fromScramS_corr = checkStats(df, scramNS, scramS, conditionNames)
    d_p = {'S_from_NS_' + variable: p_fromNS_corr,
         'NS_from_Scram_NS_' + variable: p_fromScramNS_corr,
         'S_from_Scram_S_' + variable: p_fromScramS_corr}
    df_p = pd.DataFrame(d_p, columns=list(d_p.keys()))

    figname = fignames[0] + att
    plt.figure(figname, constrained_layout=True)
    plt.suptitle(figname)
    plt.title(variable)
    
    if mainPlot == 'bar':
        ax = sns.barplot(data=df, orient="v", saturation=0.3, ci=95, capsize=0.05, errwidth=2, alpha=0.8)
        for bar, color in zip(ax.containers[0], colors):
            bar.set_color(color)
    elif mainPlot == 'violin':
        ax = sns.violinplot(data=df, orient="v", saturation=0.3, ci=95, capsize=0.05, errwidth=2, alpha=0.8)
    elif mainPlot == 'dot':
        colNames = df.columns.tolist()
        x = np.linspace(0, len(colNames) - 1, len(colNames))
        y_mean = np.nanmean(np.asarray(df, dtype=np.float64), axis=0)
        y_std = np.nanstd(np.asarray(df, dtype=np.float64), axis=0)

        leng = df.shape[0]
        n = leng - (df.isna().sum())
        y_ste = y_std / np.sqrt(n)

        for idcol, color in enumerate(colors):
            xd = x[idcol]
            yd = y_mean[idcol]
            if errMode == 'std':
                ye = y_std[idcol]
            elif errMode == 'se':
                ye = y_ste[idcol]
            elif type(errMode) == float:
                dat = np.asarray(df, dtype=np.float64)[:, idcol]
                dat = dat[~np.isnan(dat)]
                if conf_meth == 'bootstrap':
                    values = [np.random.choice(dat, size=len(x), replace=True).mean() for i in range(1000)]
                    ye = np.percentile(values, [100 * (1 - errMode) / 2, 100 * (1 - (1 - errMode) / 2)])[1] - yd
                else:
                    ye = stats.t.interval(confidence=errMode, df=n[idcol], loc=yd, scale=y_ste[idcol])[1] - yd

            colName = colNames[idcol]
            plt.scatter(xd, yd, color=color, label=colName)
            plt.plot((xd, xd), (yd - ye, yd + ye), color=color)
            plt.plot((xd - wid, xd + wid), (yd - ye, yd - ye), color=color)
            plt.plot((xd - wid, xd + wid), (yd + ye, yd + ye), color=color)

        plt.xticks(x, colNames)

    if plotType == 'strip':
        sns.stripplot(data=df, orient="v", size=2, jitter=True, dodge=True, edgecolor="white", alpha=alpha, color='gray')
        plt.xticks(rotation=45)
    elif plotType == 'swarm':
        sns.swarmplot(data=df, orient="v", size=1, alpha=alpha, color='gray')
        plt.xticks(rotation=45)

    if ylim is not None:
        plt.ylim(ylim)
    
    return df, df_p

def plot_VPI_diff(subplotNum,variable,conditionNames,NS_data,S_data,ylim=None,mainPlot='dot',plotType='swarm',fignames=['All','NS','S'],att='',size=2,baralpha=0.8,wid=0.2,errMode='se',conf_meth='',alpha=0.4):
    scramDiff=np.array(S_data[0],dtype=np.float64)-np.array(NS_data[0],dtype=np.float64)
    colors = [i for s in [[(0.75,0.75,0.75)],sns.color_palette("deep", 15)[:len(conditionNames[1:])]] for i in s]
    # colors=['gray','blue','r','g','m','y','c','darkred','gray','blue','r','g','m','y','c','darkred']
    # colors=colors[:int(len(colors)/2)]
    series_list = []
    
    for i, name in enumerate(conditionNames):
        s = pd.Series(np.array(S_data[i],dtype=np.float64)-np.array(NS_data[i],dtype=np.float64), name=variable +" " + name)
        series_list.append(s)
    
    # Do Mann Whitneys
    dfP = pd.concat(series_list, axis=1)
    p=checkStatsDiff(dfP,variable,scramDiff,conditionNames)
    plt.subplot(subplotNum[0],subplotNum[1],subplotNum[2])
    # from here edit
    if mainPlot=='bar':
        # sns.barplot(data=dfP, orient="v", saturation=0.2, color=(0.75,0.75,0.75,baralpha), ci=95, capsize=0.05, errwidth=2)
        ax=sns.barplot(data=dfP, orient="v", saturation=0.3, ci=95, capsize=0.05, errwidth=2,alpha=0.8, linewidth=0)
        for bar, color in zip(ax.containers[0], colors):
            bar.set_color(color)
    elif mainPlot=='violin':
        ax=sns.violinplot(data=dfP, orient="v", saturation=0.3, ci=95, capsize=0.05, errwidth=2,alpha=0.8, linewidth=0)
        
    elif mainPlot=='dot':
        # find x of data
        colNames=dfP.columns.tolist()
        x=np.linspace(0,len(colNames)-1,len(colNames))
        # find y postiiton of mean and std err + and -
        y_mean=np.nanmean(np.asarray(dfP),axis=0)
        y_std=np.nanstd(np.asarray(dfP),axis=0)
        
        # work out n for each column
        leng=dfP.shape[0]
        n=leng-(dfP.isna().sum())
        y_ste=y_std/np.sqrt(n)
        
        # draw points - circle for mean, line for err
        for idcol,color in enumerate(colors):
            xd=x[idcol]
            yd=y_mean[idcol]
            if errMode=='sd':
                ye=y_std[idcol]
            elif errMode=='se':
                ye=y_ste[idcol]
            elif type(errMode)==float:
                dat=np.asarray(dfP)[:,idcol]
                dat = dat[~np.isnan(dat)]
                if conf_meth=='bootstrap':
                    # compute 95% CI by 1k bootstrap...
                    values = [np.random.choice(dat,size=len(x),replace=True).mean() for i in range(1000)] 
                    ye=np.percentile(values,[100*(1-errMode)/2,100*(1-(1-errMode)/2)])[1]-yd
                else:
                    # or by t-distribution
                    ye=stats.t.interval(alpha=errMode, df=n[idcol], loc=yd, scale=y_ste[idcol])[1]-yd 
            
            colName=colNames[idcol]
            plt.scatter(xd,yd,color=color,label=colName)
            # draw line between (x,y-err):(x,y+err)
            plt.plot((xd,xd),(yd-ye,yd+ye),color=color)
            plt.plot((xd-wid,xd+wid),(yd-ye,yd-ye),color=color)
            plt.plot((xd-wid,xd+wid),(yd+ye,yd+ye),color=color)
            
        plt.xticks(x, colNames[15:])
    
    if plotType=='strip':
        plt.subplot(subplotNum[0],subplotNum[1],subplotNum[2])
        sns.stripplot(data=dfP, orient="v", size=2, jitter=True, dodge=True, edgecolor="white",alpha=alpha,color='gray')
        plt.xticks(rotation=45)
    
    elif plotType=='swarm':
        plt.subplot(subplotNum[0],subplotNum[1],subplotNum[2])
        sns.swarmplot(data=dfP, orient="v", size=1,alpha=alpha, color='gray')
        plt.xticks(rotation=45)
    
    if ylim is not None:
        plt.ylim(ylim)
        
    return dfP, p

def checkStats(df,scram_NS,scram_S,conditionNames):
    p_fromNS,p_fromScramNS,p_fromScramS=[],[],[]
    for i, name in enumerate(conditionNames):
        
        # Social phase Stat different from NonSocial phase?  
        group1=df['NS: ' + name]
        group1=group1.apply(pd.to_numeric, errors='coerce')
        group1=group1[~np.isnan(group1)]
        group2=df['S: ' + name]
        group2=group2.apply(pd.to_numeric, errors='coerce')
        group2=group2[~np.isnan(group2)]
        _,p=stats.mannwhitneyu(group1, group2, alternative='two-sided')
        p_fromNS.append(p)
        
        # NonSocial phase Stat different from NS scram?    
        group1=df['NS: ' + name]
        group1=group1.apply(pd.to_numeric, errors='coerce')
        group1=group1[~np.isnan(group1)]
        group2=np.array(scram_NS)
        group2=group2[~np.isnan(group2)]
        t,p=stats.mannwhitneyu(group1, group2, alternative='two-sided')
        p_fromScramNS.append(p)
        
        # Social phase Stat different from S scram?    
        group1=df['S: ' + name]
        group1=group1.apply(pd.to_numeric, errors='coerce')
        group1=group1[~np.isnan(group1)]
        group2=scram_S
        group2=np.array(scram_S)
        group2=group2[~np.isnan(group2)]
        t,p=stats.mannwhitneyu(group1, group2, alternative='two-sided')
        p_fromScramS.append(p)
        
    # Correct p-values for multiple comparisons using FDR
    p_fromNS_corr = multipletests(p_fromNS, method='fdr_bh')[1]
    p_fromScramNS_corr = multipletests(p_fromScramNS, method='fdr_bh')[1]
    p_fromScramS_corr = multipletests(p_fromScramS, method='fdr_bh')[1]
    
    return p_fromNS,p_fromScramNS,p_fromScramS, p_fromNS_corr,p_fromScramNS_corr,p_fromScramS_corr
#%% compute a difference dataframe between social and non-social phases of experiment
def compute_phase_difference(df):
    """
    Computes the phase difference for each metric in the input dataframe and returns a new dataframe with the
    differences.

    :param df: A pandas dataframe containing the behavioural measures for two different phases of an experiment.
               Each column is named with the suffix '_NS' or '_S' with the exception of 'Genotype'.
               These represent the two phases.
    :return: A new dataframe with additional columns for the phase differences of each metric.
    """
    metrics = ['Genotype', 'VPI', 'Freezes', 'Percent_Moving', 'Distance', 'midCrossings', 'BPS', 'MeanBoutDistance', 'MeanBoutAngle', 'propTurns']

    diff_df = pd.DataFrame()  # create a new dataframe to hold the differences

    for metric in metrics:
        if metric == 'Genotype':
            diff_df[metric] = df[metric]
        else:
            ns_col = metric + '_NS'
            s_col = metric + '_S'
            diff_col = metric + '_Diff'
    
            diff_df[diff_col] = df[s_col] - df[ns_col]

    return diff_df

#%% perform mannwhitney with bonferroni correction
def mann_whitney_all_vs_scrambled(df):
    """
    Perform Mann-Whitney U tests comparing each genotype to a genotype called 'Scrambled'
    for all columns in the input dataframe, and correct for multiple comparisons using FDR.

    Parameters:
        df (pandas.DataFrame): Input dataframe containing the behavioral data.

    Returns:
        pandas.DataFrame: Dataframe containing the p-values and FDR-corrected p-values for each genotype and column.
    """
    # Get list of genotypes
    genotypes = df["Genotype"].unique().tolist()

    # Remove "Scrambled" from list of genotypes
    genotypes.remove("Scrambled")

    # Create empty dataframe to store results
    results_df = pd.DataFrame()

    # Perform Mann-Whitney U test for each genotype compared to "Scrambled" for all columns
    for col_name in df.columns:
        if col_name == "Genotype":
            continue
        for genotype in genotypes:
            group1 = df.loc[df["Genotype"] == "Scrambled", col_name]
            group2 = df.loc[df["Genotype"] == genotype, col_name]
            _, pvalue = mannwhitneyu(group1, group2, alternative="two-sided")
            results_df = results_df.append({"Genotype": genotype, "Column": col_name, "pvalue": pvalue}, ignore_index=True)

    # Correct p-values for multiple comparisons using FDR
    results_df["pvalue_corr"] = fdrcorrection(results_df["pvalue"])[1]

    return results_df
#%%
def analyseTailTracking(XFile,Yfile,params,save=True):

        # read tailX and tailY files
        xData=np.genfromtxt(XFile, delimiter=',')
        yData=np.genfromtxt(Yfile, delimiter=',')
        [bx,by,ex,ey]=params
        
        # Determine number of frames, segments and angles
        num_frames = np.shape(xData)[0]-1
        num_segments = np.shape(xData)[1]
        num_angles = num_segments - 1
      
        # Allocate space for measurements - creates empty arrays to store data
        cumulAngles = np.zeros(num_frames)
        curvatures = np.zeros(num_frames)
        bodyTheta = np.zeros([num_frames,num_angles])
        
        # Measure tail motion, angles, and curvature ##
        ########### Start of frame loop (f) #################
        print('numFrames =' + str(num_frames))
        for f in range(num_frames):
            #print(str(f))
            delta_thetas = np.zeros(num_angles) # Make an array of zeros with the same size as num angles (num seg-1)
           
            # find angle between eyes and body (heading again)
            dx = bx[f] - ex[f]  
            dy = by[f] - ey[f] 
            heading = np.arctan2(dy, dx) * 360.0 / (2.0*np.pi)
           
            prev_theta = heading # set first theta to heading
            # measure angle between body and last segment
            dx = xData[f, -1] - bx[f] 
            dy = yData[f, -1] - by[f]
            theta = np.arctan2(dy, dx) * 360.0 / (2.0*np.pi) # calc arctangent bt dx and dy, convert to deg
            if np.abs(theta - prev_theta)>180:
                theta*=-1
            ############### Start of angle loop (a) ################
            for a in range(num_angles):
                ddx = xData[f, a] - bx[f]
                ddy = yData[f, a] - by[f]
                thetaa = np.arctan2(ddy, ddx) * 360.0 / (2.0*np.pi) # calc arctangent bt dx and dy, convert to deg
                if np.abs(thetaa - heading)>180:
                    thetaa*=-1
                bodyTheta[f, a] = thetaa - heading
               
                dx = xData[f, a+1] - xData[f, a] # dx between each segment for the same frame
                dy = yData[f, a+1] - yData[f, a] # dy between each segment for the same frame
                theta = np.arctan2(dy, dx) * 360.0 / (2.0*np.pi) # calc arctangent bt dx and dy, convert to deg
                if np.abs(theta - prev_theta)>180:
                    theta*=-1
                delta_thetas[a] = theta - prev_theta
                prev_theta = theta # prev theta is set to current theta
            ############### End of angle loop (a) ################
           
            cumulAngles[f] = np.sum(delta_thetas) # sum all angles for this frame
            curvatures[f] = np.mean(np.abs(delta_thetas)) # mean of abs value of angles
            
        ########### End of frame loop (f) ###############
        tail=np.zeros([num_frames,2])
        tail[:,0]=cumulAngles
        tail[:,1]=curvatures
            
        return tail,bodyTheta


#  rotate all points in a list of trajectories (x and y seperately) by the initial heading
def rotateTrajectoriesByHeadings(trajectoriesX,trajectoriesY,trajectoriesHeadings):
    
    rotTrajectoriesX=[]
    rotTrajectoriesY=[]
    
    numTraj=len(trajectoriesX)
    for i in range(0,numTraj):
        trajX=trajectoriesX[i]
        trajY=trajectoriesY[i]
        trajHeading=trajectoriesHeadings[i]
        lenTraj=len(trajX)
        rotTrajX=[]
        rotTrajY=[]
        
        for j in range(0,lenTraj):
            qx,qy=SCZM.rotatePointAboutOrigin(trajX[j],trajY[j],trajHeading)
            rotTrajX.append(qx*-1)
            rotTrajY.append(qy*-1)
            
        rotTrajectoriesX.append(rotTrajX)
        rotTrajectoriesY.append(rotTrajY)
        
    return rotTrajectoriesX,rotTrajectoriesY

# extracts trajectories from tracking data according to stimulus protocol
def extractTrajFromStim(loomStarts,loomEnds,fx,fy,heading):
    
    trajectoriesHeadings=[]
    trajectoriesX=[]
    trajectoriesY=[]
    txx=[]
    tyy=[]
    
    for i,start in enumerate(loomStarts):
        end=loomEnds[i]
        tx=fx[start:end]
        txx.append(tx)
        trajX=list(map((lambda x: x - fx[start]), tx)) # subtract the starting x from this trajectory
        trajectoriesX.append(trajX)    
        
        ty=fy[start:end]
        tyy.append(ty)
        trajY=list(map((lambda x: x - fy[start]), ty)) # subtract the starting y from this trajectory
        trajectoriesY.append(trajY)
        
        trajectoriesHeadings.append(heading[start])
        
    
    return txx,tyy,trajectoriesHeadings,trajectoriesX,trajectoriesY



def extractVecFromStim(loomStarts,loomEnds,vecIn):
# segments and extracts any vector according to stimulus protocol
    
    matOut=[]
    
    for i,start in enumerate(loomStarts):
        end=loomEnds[i]
        matOut.append(vecIn[start:end])
    
    return matOut

def findLooms(movieLengthFr,startTime=15,interval=2,duration=1,numSecsAnalyse=2,frameRate=100,responseSeconds=0.5):
# Finds loom start and end positions in frames from input stimulus protocol parameters
# startTime in minutes, length of adaptation
# interval in minutes, time between looms
# duration in seconds, duration of stim (since stimulus ENDS on the marker time, so starts marker-duration)
# movieLengthFr is length of full movie in frames including adaptation 
# numFrames is the seconds to extract for trajectory analysis
# frameRate is camera frame rate in Hz
# returns lists of loomStart and end positions frame positions    
    latencyLimitFr=np.int(responseSeconds*frameRate)
    startFrame=np.int(startTime*60*frameRate)
    intervalFrames=np.int(interval*60*frameRate)
    durationFrames=np.int(duration*frameRate)
    firstLoomStart=startFrame+intervalFrames-durationFrames
    loomStarts=[]
    loomEnds=[]
    respLoomEnds=[]
    loomStarts.append(firstLoomStart)
    numFrames=np.int(numSecsAnalyse*frameRate)
    loomEnds.append(firstLoomStart+numFrames-1)
    respLoomEnds.append(firstLoomStart+latencyLimitFr-1)
    
    numLooms=np.int(np.floor((movieLengthFr-startFrame)/intervalFrames))
    rem=False
    for i in range(1,numLooms):
        loomStarts.append(loomStarts[i-1]+intervalFrames)
        loomEnds.append(loomEnds[i-1]+intervalFrames)
        respLoomEnds.append(respLoomEnds[i-1]+intervalFrames)
        if loomEnds[i]>movieLengthFr or respLoomEnds[i]>movieLengthFr:
            loomStarts[i]='xxx'
            loomEnds[i]='xxx'
            respLoomEnds[i]='xxx'
            rem=True
            break
    if rem:
        respLoomEnds.remove('xxx')
        loomStarts.remove('xxx')
        loomEnds.remove('xxx')
    
    
    
    
    return loomStarts,loomEnds,respLoomEnds

## Dispersal analysis: take trajectory in 5 second windows and find the smallest circle that encompasses that trajectory. 
## Needs only fx, fy and framerate
def measureDispersal(xPos,yPos,frameRate=120,window=5):
    print('Measuring dispersal...')
    winFr=np.int(np.floor(window*frameRate))
    numFr=len(xPos)
    dispVec=np.zeros(numFr)
    
    for i in range(numFr): # cycle through all time points
#        print(i)
        if i>winFr: # skip first window
            if i==np.floor_divide(numFr,2): print('Halfway through dispersal measurement...')
            xPosj=[]
            yPosj=[]
            for j in range(i-winFr,i): # cycle through positions window secs behind
                # build this trajectory
                xPosj.append(xPos[j])
                yPosj.append(yPos[j])
            
            # find time points of most distant points in x and y
            x1_t=np.where(xPosj==np.min(xPosj))[0][0]
            x2_t=np.where(xPosj==np.max(xPosj))[0][0]
            y1_t=np.where(yPosj==np.min(yPosj))[0][0]
            y2_t=np.where(yPosj==np.max(yPosj))[0][0]
            
            # find their coordinates
            x1=([xPosj[x1_t],yPosj[x1_t]])
            x2=([xPosj[x2_t],yPosj[x2_t]])
            y1=([xPosj[y1_t],yPosj[y1_t]])
            y2=([xPosj[y2_t],yPosj[y2_t]])
            
            # find distance between each of these points and find largest
            xy=[]
            xy.append(SCZU.computeDist(x1[0],x1[1],x2[0],x2[1]))
            xy.append(SCZU.computeDist(x1[0],x1[1],y1[0],y1[1]))
            xy.append(SCZU.computeDist(x1[0],x1[1],y2[0],y2[1]))
            xy.append(SCZU.computeDist(x2[0],x2[1],y1[0],y1[1]))
            xy.append(SCZU.computeDist(x2[0],x2[1],y2[0],y2[1]))
            xy.append(SCZU.computeDist(y1[0],y1[1],y2[0],y2[1]))
            dispVec[i] = np.max(xy)/2 # take half the largest distance as radius for this time frame
    
    # pad the start of vector (first window seconds) with the first dispersal value 
    for i in range(winFr):
        dispVec[0:winFr]=dispVec[winFr+1]
        
    print('Finished measuring dispersal')
    return dispVec
#    # (OPTIONAL) Pick a part of the trajectory to display and draw a circle for presentations
#    if(plotCirc):
#        
#        CircMed_t=np.where(dispVec==np.median(dispVec))[0][0]
#        CircMax_t=np.where(dispVec==np.max(dispVec))[0][0]
#        
#        medRad=dispVec[CircMed_t]
#        maxRad=dispVec[CircMax_t]
#        
#        medx=xPos[CircMed_t]
#        medy=yPos[CircMed_t]
#        
#        maxx=xPos[CircMax_t]
#        maxy=yPos[CircMax_t]
#        
#        # grab trajectories
#        medxPosj=[]
#        medyPosj=[]
#        maxxPosj=[]
#        maxyPosj=[]
#        for j in range(CircMed_t-winFr,CircMed_t): # cycle through positions window secs behind
#            # build this trajectory
#            medxPosj.append(xPos[j])
#            medyPosj.append(yPos[j])
#        
#        for j in range(CircMax_t-winFr,CircMax_t): # cycle through positions window secs behind
#            # build this trajectory
#            maxxPosj.append(xPos[j])
#            maxyPosj.append(yPos[j])
#            
#        figName='MedTrajCirc'
#        plt.figure(figName)
#        plt.title(figName)
#        plt.plot(medxPosj,medyPosj)
#        H=(np.median(dispVec)*2)+5 # mm
#        lim=(np.sqrt(np.square(H)/2))
##        xst=np.min(medxPosj)-2.5
##        yst=np.min(medyPosj)-2.5
##        plt.xlim(xst,xst+lim)
##        plt.ylim(yst,yst+lim)
#        ax = plt.gca()
#        ylims=ax.get_ylim()
#        xlims=ax.get_xlim()
#        Xcen=np.mean(xlims)
#        Ycen=np.mean(ylims)
#        plt.xlim((Xcen-(lim/2)),Xcen+(lim/2))
#        plt.ylim((Ycen-(lim/2)),Ycen+(lim/2))
#        circlemed=plt.Circle((np.mean(xlims),np.mean(ylims)),medRad,color='blue',fill=False)
#        ax.add_artist(circlemed)
#        
#        figName='MaxTrajCirc'
#        plt.figure(figName)
#        plt.title(figName)
#        plt.plot(maxxPosj,maxyPosj)
#        H=(np.max(dispVec)*2)+5 # mm
#        lim=(np.sqrt(np.square(H)/2))
#        ax = plt.gca()
#        ylims=ax.get_ylim()
#        xlims=ax.get_xlim()
#        Xcen=np.mean(xlims)
#        Ycen=np.mean(ylims)
#        plt.xlim((Xcen-(lim/2)),Xcen+(lim/2))
#        plt.ylim((Ycen-(lim/2)),Ycen+(lim/2))
#        circlemax=plt.Circle((np.mean(xlims),np.mean(ylims)),maxRad,color='blue',fill=False)
#        ax.add_artist(circlemax)
    
   

def findBoutsXY(fx,fy,boutStarts):
    
    boutX = np.zeros(len(boutStarts))
    boutY = np.zeros(len(boutStarts))
    
    # cycle through boutStarts
    for i,s in enumerate(boutStarts):
        boutX[i]=fx[s]
        boutY[i]=fy[s]
#    print('Done')
    return boutX,boutY

def computeTimeCentreSurround(fx,fy,imageWidth=1280,imageHeight=1280,gridSize=40): # position of centre is hardcoded = only works for 20,20 heatmaps (default)
    
    ## this is the hard coded bit that means it only works with 40 x 40 gridSize at present
    startXH= 8
    startYH= 8
    endXH= 31
    endYH = 31
    
    locs=[]
    locs.append('Centre')
    locs.append('Surround')
    
    # create matrixs of zeros the same size as gridSize in hieght and width: boutCount, featureSum 
    startX= np.zeros(gridSize)
    startY= np.zeros(gridSize)
    endX= np.zeros(gridSize)
    endY = np.zeros(gridSize)
    
    for i in range(len(startX)): # cycle through i = startX
        div = imageWidth / gridSize # divide imageWidth by gridSize, 
        startX[i] = div * i 
        endX[i] = div * (i+1) 
        
        div = imageHeight / gridSize
        startY[i] = div * i
        endY[i] = div * (i+1) 
        
    pntCountCentre=0
    pntCountSurround=0
    for j in range(len(fx)): # cycle through all frames
        
        gridX = -1
        gridY = -1
        
        ## Is x position in centre?
        for k in range(len(locs)): # cycle through gridXLocations
            if fx[j] >startX[startXH] and fx[j]<endX[endXH]: # then we are in the central range of x
                gridX=k
                # Is y position in centre?
                for l in range(len(locs)): # cycle though gridYLocations
                    if fy[j] >startY[startYH] and fy[j] <endY[endYH]: # then we are in the central range of y
                        gridY=l
                        break
                
        if gridX>=0 and gridY>=0: # check we found this frame
            pntCountCentre += 1 # update count grid with location of this bout
        else: 
            pntCountSurround += 1
            
    pntCentreNorm=pntCountCentre/len(fx)
    pntSurroundNorm=pntCountSurround/len(fx)
    
    return pntCentreNorm,pntSurroundNorm
            
def heatmapFeature(fx,fy,boutStarts,feature,imageWidth=[],imageHeight=[],gridSize=20):
    
    # create matrixs of zeros the same size as gridSize in hieght and width: boutCount, featureSum 
    boutCount = np.zeros([gridSize,gridSize])
    featureSum = np.zeros([gridSize,gridSize])
    
    startX= np.zeros(gridSize)
    startY= np.zeros(gridSize)
    endX= np.zeros(gridSize)
    endY = np.zeros(gridSize)
    lostBoutsCount=0
    
    for i in range(len(startX)): # cycle through i = startX
        div = imageWidth / gridSize # divide imageWidth by gridSize, 
        startX[i] = div * i 
        endX[i] = div * (i+1) 
        
        div = imageHeight / gridSize
        startY[i] = div * i
        endY[i] = div * (i+1) 
    
    xBout,yBout = findBoutsXY(fx,fy,boutStarts)
    
    for j in range(len(xBout)): # cycle through bouts
        
        gridX = -1
        gridY = -1
        ## Find X position in grid for this bout
        for k in range(gridSize): # cycle through gridXLocations
            if xBout[j] >startX[k] and xBout[j]<endX[k]: # then we are in the right x
                gridX=k
                # Find y position
                for l in range(gridSize): # cycle though gridYLocations
                    if yBout[j] >startY[l] and yBout[j] <endY[l]: # then we are in the right y
                        gridY=l
                        break
                
        if gridX>=0 and gridY>=0: # check we found this bout
            boutCount[gridX,gridY] += 1 # update boutCount grid with location of this bout
            featureSum[gridX,gridY]=featureSum[gridX,gridY]+feature[j]
        else: 
            lostBoutsCount+=1
            
    ##  grids to PDF and means
    np.seterr(divide='ignore',invalid='ignore')
    boutDensity=np.divide(boutCount,np.sum(boutCount))
    featureDensity=np.divide(featureSum,np.sum(featureSum))
    featureAverage=np.divide(featureSum,boutCount)
    featureDensity[np.isnan(featureDensity)]=0
    featureAverage[np.isnan(featureAverage)]=0
    print('Lost ' + str(lostBoutsCount) + ' frames.')
    return boutCount,boutDensity,featureSum,featureDensity,featureAverage
    
def extractBouts(fx,fy, ort, distPerSec,name=[],savepath=[],FPS=120,plot=False, preWindow=100, postWindow=300,save=True):
    ret=1 # assume it works
    
    ## Compute activity level of the fish in bouts per second (BPS)
    preWindow=int(np.floor((preWindow/1000)*FPS))
    postWindow=int(np.floor((postWindow/1000)*FPS))
    _,motion_signal=SCZU.motion_signal(fx,fy,ort)
    # Find bouts starts and stops
    boutStarts = []
    boutStops = []
    moving = 0
    dM=np.zeros(len(motion_signal))
    dM[1:]=ndimage.filters.gaussian_filter1d(np.diff(motion_signal),2)

    # thresholds should be computed for each movie. Plot out a histogram of the deltas in motion_signal. We assume the noise is Gaussian, so fit a Gaussian to the data. 5* the Sigma of this Gaussian should be a good threshold.
#    sortDM=sorted(np.abs(dM))
#    sampleDM=sortDM[0:int(np.floor(len(sortDM)*0.8))] # remove extreme 10% of values (i.e. ignore the ridiculously high motion signals as we want to estimate the noise)
    histDM,c=np.histogram(dM, bins=500) # plot it out in a histogram
    cc = (c[:-1]+c[1:])/2
    mp=np.max(histDM)
    sd=np.std(dM)
    initial_guess = [mp,0,sd]
    popt, pcov = curve_fit(SCZU.gaussian_func, cc, histDM,p0=initial_guess)
    if plot:
        plt.scatter(cc,histDM)
        plt.plot(cc,SCZU.gaussian_func(cc,*popt))
        plt.show() 
    
    # in order to be counted as a bout, the rate of increase must be at least 5 sigmas from the gaussian: appears robust when used on a slightly smoothed trace     
    # in order for the bout to end, it must fall to below 1 sigma for 5 consecutive frames and be at least 40 frames after the initiation of the bout (the latter probably not needed)
    startThreshold = popt[2]*4
    print('start thresh is ' + str(startThreshold))
    stopThreshold = popt[2]
    print('stop thresh is' + str(stopThreshold))
    smooth=SCZU.smoothSignal(dM,5)
    for i, m in enumerate(smooth):
        br=0
        if(moving == 0):
            if br==1:
                break
            if m > startThreshold:
                moving = 1
                boutStarts.append(i)
                ii=i
        else:
            if np.abs(m)<stopThreshold and i>ii+40:
                for k in range(i,i+5):
                    if k < len(dM):
                        if np.abs(dM[k]) > stopThreshold:
                            moving = 1
                            break
                        else:
                            moving = 0
                    else:
                        break
    
    # Extract all bouts (ignore last and or first, if clipped)
    boutStarts = np.array(boutStarts)
    for i in range(0,len(boutStarts)):
        boutStops.append((boutStarts[i]-preWindow)+postWindow)
    boutStops = np.array(boutStops)
    
    while(len(boutStarts) > len(boutStops)):
        boutStarts = boutStarts[:-1]
    while(len(boutStops) > len(boutStarts)):
        boutStops = boutStops[:-1]
    if len(boutStarts)!=0:
        if(boutStarts[0]<preWindow+1):
            boutStarts=boutStarts[1:] 
            boutStops=boutStops[1:]        
    if len(boutStarts)!=0:
        if(boutStarts[len(boutStarts)-1]+postWindow>len(motion_signal)):
            boutStarts=boutStarts[:-1]
            boutStops=boutStops[:-1]
        
    # Return if failed to find any unclipped bouts
    if(len(boutStarts)==0 or len(boutStops)==0): 
        return -1,-1, -1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1
    # threshold on maximum
#    keep = np.zeros(len(boutStarts))
#    for i,b in enumerate(boutStarts):
#        mm=np.max(AZM.smoothSignal(motion_signal[b-preWindow:b+postWindow],5))>1.2
#        if mm: keep[i]=1
#    keep=keep!=0
#    boutStarts=boutStarts[keep]
#    boutStops=boutStops[keep]
    
    # Count number of bouts
    numBouts= len(boutStarts)
    numberOfSeconds = np.size(motion_signal)/FPS  

    # Set the bouts per second (BPS)
    boutsPerSecond = numBouts/numberOfSeconds
    
    # Extract the bouts; distance, motion_signal, orientation and angles
    boutStarts = boutStarts[(boutStarts > preWindow) * (boutStarts < (len(motion_signal)-postWindow))]

    allBoutsDist = np.zeros([len(boutStarts), (preWindow+postWindow)])    
    allBouts = np.zeros([len(boutStarts), (preWindow+postWindow)])
    allBoutsOrt = np.zeros([len(boutStarts), (preWindow+postWindow)])
    boutAngles = np.zeros(len(boutStarts))
    
    for b in range(0,len(boutStarts)):
        allBoutsDist[b,:] = distPerSec[(boutStarts[b]-preWindow):(boutStarts[b]+postWindow)] # extract velocity over this bout
        allBouts[b,:] = motion_signal[(boutStarts[b]-preWindow):(boutStarts[b]+postWindow)] # extract motion over this bout
        allBoutsOrt[b,:] = rotateOrt(ort[(boutStarts[b]-preWindow):(boutStarts[b]+postWindow)]) # extract heading for this bout (rotated to zero initial heading)
        boutAngles[b] = np.mean(allBoutsOrt[b,-2:-1])-np.mean(allBoutsOrt[b,0:1]) # take the heading before and after

    Lturns=np.zeros(len(boutAngles))    
    Rturns=np.zeros(len(boutAngles))    
    FSwims=np.zeros(len(boutAngles))
    
    for i,angle in enumerate(boutAngles):
        if angle > 10: 
            Lturns[i]=1
        elif angle < -10:
            Rturns[i]=1
        else:
            FSwims[i]=1
    Rturns=Rturns!=0
    Lturns=Lturns!=0
    FSwims=FSwims!=0
    LturnPC=(np.sum(Lturns)/(np.sum(Lturns)+np.sum(Rturns)))*100
    
    if plot:
        plt.figure('Bout finder Analysis First 30 seconds')
        xFr=range(len(motion_signal))
        x=np.divide(xFr,FPS)
        ef=FPS*30
        plt.plot(x[:ef],motion_signal[:ef])
        boutStartMarker=np.zeros(len(motion_signal[:ef]))
        boutMarker=np.zeros(len(motion_signal[:ef]))
        for i in range(len(boutStarts[boutStarts<(ef-postWindow)])):
            boutStartMarker[boutStarts[i]]=10
            boutMarker[(boutStarts[i]-preWindow):(boutStarts[i]+postWindow)]=10
    
        plt.plot(x[:ef],boutStartMarker,'-r',linewidth=0.5)
#        plt.plot(x,boutMarker)
        plt.fill_between(x[:ef],boutMarker,0,alpha=0.4,color='Orange')
        figName='Bout finder Analysis'
        plt.title(figName)
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (mm/sec)')
        
        if plot and save:
            savepath=savepath+ 'BoutFinder/'
            SCZU.cycleMkDir(savepath)
            saveName=savepath + name+'_BoutFinder.png'
            plt.savefig(saveName,dpi=600)
            
    while(len(boutStarts) > len(boutStops)):
        boutStarts = boutStarts[:-1]
    while(len(boutStops) > len(boutStarts)):
        boutStops = boutStops[:-1]
        
    return ret,Rturns,Lturns,FSwims,boutsPerSecond, allBouts, allBoutsDist, allBoutsOrt, boutAngles, LturnPC, boutStarts, boutStops, motion_signal, dM
  
# rotate the orientation trace so that the initial heading is zero
def rotateOrt(ort):
    ort_rot=np.zeros(len(ort))
    ort_init=ort[0]
    for i,thisOrt in enumerate(ort):
        o=thisOrt-ort_init
        oAbs=np.abs(o)
        if o>180:
            o=(180-(o-180))*-1
        elif o<-180:
            o=180-(oAbs-180)
            
        ort_rot[i]=o
    return ort_rot
        
        

## compute cumulative angle from heading
def computeCumulativeAngle(ort,plot=True,FPS=120):
    # rotate heading so starting heading is zero
    ortRot=rotateOrt(ort)
    
    ortDiff=np.diff(ortRot)# differentiate the headings
    ortDiff=SCZU.filterTrackingFlips(ortDiff) # check for unlikely flips in the tracking caused by blurred eyes during movement (assumes turns of more than 100 degrees are impossible... which they are not)
    ortDiffAbs=np.abs(ortDiff) # take the absolute
    ort2pi=360-ortDiffAbs # subtract a circle from the absolute difference
    
    # we will now compare ort2pi and ortDiffAbs to accumulate the smallest in magnitude but keep the sign. 
    cumOrt=np.zeros(len(ortDiff)-1)
    for i in range(len(cumOrt)):
        if i !=0: 
           if ort2pi[i]<ortDiffAbs[i]:
               cumOrt[i]=ort2pi[i]+cumOrt[i-1]
           else:
               cumOrt[i]=ortDiff[i]+cumOrt[i-1]
    ortAbs=SCZU.accumulate(ortDiffAbs)
    totAngle=ortAbs[-1]               
    cumAngle=cumOrt[-1]
    avAngVel=cumAngle/((len(cumOrt))/FPS)
    bias=cumAngle/totAngle # positive is left turns, negative is right turns

    if(plot):
        xFr=range(len(ort))
        x=np.divide(xFr,FPS)
        plt.figure()
        plt.plot(x,cumOrt)
        plt.title('Cumulative angle')
        plt.xlabel('Time (s)')
        plt.ylabel('Cumulative angle (degrees from initial heading)')

    return avAngVel,bias,cumOrt


def unpackGroupDictFile(dicFileName):
    dic=np.load(dicFileName,allow_pickle=True).item()
    met=dic['Metrics']
    avg=dic['avgData']
    name=dic['Name']
    avgCumDistAV=met['cumDist']['Mean']
    avgCumDistSEM=met['cumDist']['SEM']
    avgBoutAV=met['avgBout']['Mean']
    avgBoutSEM=met['avgBout']['SEM']
    avgHeatmap=met['avgHeatmap']['Mean']
    avgVelocity=avg['avgVelocity']
    avgAngVelocityBouts=avg['avgAngVelocityBout']
    biasLeftBout=avg['biasLeftBout']
    LTurnPC=avg['LTurnPC']
    avgBoutAmps=avg['boutAmps']
    allBPS=avg['BPSs']
    
    return name,avgCumDistAV,avgCumDistSEM,avgBoutAmps,allBPS,avgBoutAV,avgBoutSEM,avgHeatmap,avgVelocity,avgAngVelocityBouts,biasLeftBout,LTurnPC    

def poolFishDict(dictList=[]):####NOT FINISHED OR REALLY NEEDED AT THIS POINT
    
    if(len(dictList)==0):
        print("Please provide a list of dictionaries to analyse (filepaths or dictionaries themselves)")
        return -1
    
    if(isinstance(dictList[0],str)):
        dictNameList=dictList
        dictList=[]
        for i,dicName in enumerate(dictNameList):
            dictList[i]=np.load(dicName,allow_pickle=True)
        
    elif(isinstance(dictList[0],dict)==False):
        print("Please provide a list of dictionaries to analyse (filepaths or dictionaries themselves)")
    
###############################################################################    
# Compute Viewing Preference Index
def computeVPI(xPositions, yPositions, testROI, stimROI, FPS=120):
     
    # Find thresholds of Y from the Test ROI in order to define the Visible area
    visiblePositionThreshold_Y = testROI[1]+(testROI[3]/2) 
    
    # Define which frames are "VISIBLE" in Y by comparing Y with the above calculated threshold
    socialTop = stimROI[1] < visiblePositionThreshold_Y
        
    # Check Y threshold 
    if socialTop:  
        AllVisibleFrames = yPositions < visiblePositionThreshold_Y   # True/False array, BUT "Zero Y" remember is located conventionally at the TOP   
    else:
        AllVisibleFrames = yPositions > visiblePositionThreshold_Y   # Opposite True/False array
    
    # Determine Non-Visible Fames
    AllNonVisibleFrames = np.logical_not(AllVisibleFrames)
   
    # Count Visible and Non-Visible Frames
    numVisibleFrames = float(np.sum(AllVisibleFrames))     # Final Sum of Visible Frames
    numNonVisibleFrames= float(np.sum(AllNonVisibleFrames))  # Final Sum of NON Visible Frames 
        
    # Compute VPI
    VPI = (numVisibleFrames-numNonVisibleFrames)/np.size(yPositions)

    # Determine number of frames in a five minute bin
    bin_size = 60 * FPS
    max_frame = bin_size * 15

    # Compute "binned" VPI
    if len(AllVisibleFrames) >= max_frame:
        visible_bins = np.sum(np.reshape(AllVisibleFrames[:max_frame].T, (bin_size, -1), order='F'), 0)
        non_visible_bins = np.sum(np.reshape(AllNonVisibleFrames[:max_frame].T, (bin_size, -1), order='F'), 0)
        VPI_bins = (visible_bins - non_visible_bins)/bin_size
    else:
        VPI_bins = np.empty(15) * np.nan

    return VPI, AllVisibleFrames, AllNonVisibleFrames, VPI_bins


# Compute Social Preference Index
def computeSPI(xPositions, yPositions, testROI, stimROI):
    
    # Find thresholds of X and Y Test ROI in order to define the social area to calculate the SPI
    socialPositionThreshold_X = testROI[0]+(testROI[2]/2)
    socialPositionThreshold_Y = testROI[1]+(testROI[3]/2) 
    
    # Define which frames are "social" in X and Y by comparing Tracking X and Y with the above calculated threshold
    socialTop = stimROI[1]<socialPositionThreshold_Y
    socialLeft = stimROI[0]<socialPositionThreshold_X
        
    # Compute Social Frames (depending on where the stimulus fish is (top or bottom))
    
   # Check X threshold 
    if socialLeft:
        
        AllSocialFrames_X_TF= xPositions<socialPositionThreshold_X    # True/False array
        
    else:
       AllSocialFrames_X_TF= xPositions>socialPositionThreshold_X   # Opposite True/False array
    
    # Check Y threshold 
    if socialTop:
        
        AllSocialFrames_Y_TF= yPositions<socialPositionThreshold_Y   # True/False array, BUT "Zero Y" remember is located conventionally at the TOP
        
           
    else:
        AllSocialFrames_Y_TF= yPositions>socialPositionThreshold_Y   # Opposite True/False array
    
        
    AllSocialFrames_TF= np.logical_and(AllSocialFrames_X_TF,AllSocialFrames_Y_TF)  # Final SOCIAL True/False array
    AllNONSocialFrames_TF=np.logical_and(AllSocialFrames_X_TF, np.logical_not(AllSocialFrames_Y_TF))   # Final NON SOCIAL True/False array
    
    # Count Social and Non-Social Frames
    numSocialFrames = float(np.sum(AllSocialFrames_TF))     # Final Sum of Social Frames
    numNONSocialFrames= float(np.sum(AllNONSocialFrames_TF))  # Final Sum of NON Social Frames 
        
    # Compute SPI
    if (numSocialFrames+numNONSocialFrames) == 0:
        SPI = 0.0
    else:
#        SPI = (numSocialFrames-numNONSocialFrames)/(numSocialFrames+numNONSocialFrames)
        SPI = (numSocialFrames-numNONSocialFrames)/np.size(yPositions)
    
    
    return SPI, AllSocialFrames_TF, AllNONSocialFrames_TF

# Compute normalized arena coordinates
def normalized_arena_coords(xPositions, yPositions, testROI, stimROI):
    
    # Rescale by chamber dimensions
    chamber_x_min = testROI[0]
    chamber_y_min = testROI[1]
    chamber_width_pixels = testROI[2]
    chamber_height_pixels = testROI[3]
    chamber_width_mm = 17
    chamber_height_mm = 42
    
    # Find thresholds of X and Y Test ROI in order to define the social area to calculate the SPI
    socialPositionThreshold_X = testROI[0]+(testROI[2]/2)
    socialPositionThreshold_Y = testROI[1]+(testROI[3]/2) 
    
    # Define which frames are "social" in X and Y by comparing Tracking X and Y with the above calculated threshold
    socialTop = stimROI[1]<socialPositionThreshold_Y
    socialLeft = stimROI[0]<socialPositionThreshold_X
        
    # Normalize positions to "social arena" coordinates (in mm): -X, -Y away from stim, +X, +Y towards stim
    norm_x = ((xPositions-chamber_x_min)/chamber_width_pixels) * chamber_width_mm
    norm_y = ((yPositions-chamber_y_min)/chamber_height_pixels) * chamber_height_mm
    
   # Place stim fish on right 
    if socialLeft:
        norm_x = (norm_x * -1) + chamber_width_mm
            
    # Place stim fish on top 
    if ~socialTop:
        norm_y = (norm_y * -1) + chamber_height_mm
        
    return norm_x, norm_y


def computeSPI_3fish(xPositions, yPositions, testROI, stimLeft):
    
    # Find thresholds of X and Y Test ROI in order to define the social area to calculate the SPI
    socialPositionThreshold_X = testROI[0]+(testROI[2]/2)
#   
        
    # Compute Social Frames (depending on where the stimulus fish is (top or bottom))
    
   # If stimLeft is TRUE
    if stimLeft:
        AllSocialFrames_X_TF= xPositions<socialPositionThreshold_X    # Social Left     
    else:
       AllSocialFrames_X_TF= xPositions>socialPositionThreshold_X   # Opposite True/False array
    
    AllNONSocialFrames_X_TF=np.logical_not(AllSocialFrames_X_TF)   # Final NON SOCIAL True/False array

    # Count Social and Non-Social Frames
    numSocialFrames = float(np.sum(AllSocialFrames_X_TF))     # Final Sum of Social Frames
    numNONSocialFrames= float(np.sum(AllNONSocialFrames_X_TF))  # Final Sum of NON Social Frames 
        
    # Compute SPI
    if (numSocialFrames+numNONSocialFrames) == 0:
        SPI = 0.0
    else:
        SPI = (numSocialFrames-numNONSocialFrames)/(numSocialFrames+numNONSocialFrames)
#        SPI = (numSocialFrames-numNONSocialFrames)/np.size(yPositions)
    
    
    return SPI, AllSocialFrames_X_TF, AllNONSocialFrames_X_TF


# Compute VISIBLE Frames (when the Test Fish could potentially see the Stim Fish)
def computeVISIBLE(xPositions, yPositions, testROI, stimROI):
    
    # Find thresholds of Y from the Test ROI in order to define the Visible area
    visiblePositionThreshold_Y = testROI[1]+(testROI[3]/2) 
    
    # Define which frames are "VISIBLE" in Y by comparing Y with the above calculated threshold
    socialTop = stimROI[1]<visiblePositionThreshold_Y
        
    # Check Y threshold 
    if socialTop:  
        AllVisibleFrames_Y_TF= yPositions<visiblePositionThreshold_Y   # True/False array, BUT "Zero Y" remember is located conventionally at the TOP   
    else:
        AllVisibleFrames_Y_TF= yPositions>visiblePositionThreshold_Y   # Opposite True/False array
    
    
    return AllVisibleFrames_Y_TF

def findBoutArea(allBouts, FPS=120):
    
    areaBouts=[]

    for i,bout in enumerate(allBouts):
#        print("I'm here on bout number " + str(i))
        plt.plot(bout)
        areaBouts.append(np.trapz(bout-np.min(bout), dx=1/FPS))
        
    return areaBouts
    
def findBoutMax(allBouts):
    
    ampBouts=np.zeros(len(allBouts))
    for i in range(len(allBouts)):
        ampBouts[i]=(np.max(allBouts[i]))
        
    return ampBouts

# Measure distance traveled during experiment (in mm)
def distance_traveled(bx, by, ROI):

    # Rescale by chamber dimensions
    chamber_width_pixels = ROI[2]
    chamber_height_pixels = ROI[3]
    chamber_width_mm = 17
    chamber_height_mm = 42
    
    # Sample position every 10 frames (10 Hz) and accumulate distance swum
    # - Only add increments greater than 0.5 mm
    num_frames = len(bx)
    prev_x = bx[0]
    prev_y = by[0]
    distance = 0
    for f in range(9,num_frames,10):
        dx = ((bx[f]-prev_x)/chamber_width_pixels) * chamber_width_mm
        dy = ((by[f]-prev_y)/chamber_height_pixels) * chamber_height_mm
        d = np.sqrt(dx*dx + dy*dy)
        if(d > 0.5):
            distance = distance + d
            prev_x = bx[f]
            prev_y = by[f]           
    
    return distance    

# Analyze Correlations between Test and Stimulus Fish
def analyze_tracking_SPI(folder, fishNumber, testROIs, stimROIs):
    
            # Analyze Tacking in Folder based on ROIs
            trackingFile = folder + r'/tracking' + str(fishNumber) + '.npz'    
            data = np.load(trackingFile)
            tracking = data['tracking']
            
            fx = tracking[:,0]      # Fish X (Centroid of binary particle)
            fy = tracking[:,1]      # Fish Y (centroid of binary particle)
            bx = tracking[:,2]      # Body X
            by = tracking[:,3]      # Body Y
            ex = tracking[:,4]      # Eye X
            ey = tracking[:,5]      # Eye Y
            area = tracking[:,6]    
            ort = tracking[:,7]     # Orientation
            motion = tracking[:,8]  # Frame-by-frame difference in particle
            
            # Compute SPI (NS)
            SPI, AllSocialFrames, AllNONSocialFrames = computeSPI(bx, by, testROIs[fishNumber-1], stimROIs[fishNumber-1])
            
            return SPI, AllSocialFrames, AllNONSocialFrames
            
# Analyze Correlations between Test and Stimulus Fish
def analyze_tracking_VISIBLE(folder, fishNumber, testROIs, stimROIs):
    
            # Analyze Tacking in Folder based on ROIs
            trackingFile = folder + r'/tracking' + str(fishNumber) + '.npz'    
            data = np.load(trackingFile)
            tracking = data['tracking']
            
            fx = tracking[:,0]      # Fish X (Centroid of binary particle)
            fy = tracking[:,1]      # Fish Y (centroid of binary particle)
            bx = tracking[:,2]      # Body X
            by = tracking[:,3]      # Body Y
            ex = tracking[:,4]      # Eye X
            ey = tracking[:,5]      # Eye Y
            area = tracking[:,6]    
            ort = tracking[:,7]     # Orientation
            motion = tracking[:,8]  # Frame-by-frame difference in particle
            
            # Compute AllVisibleFrames_Y_TFisible Frames
            AllVisibleFrames_Y_TF = computeVISIBLE(bx, by, testROIs[fishNumber-1], stimROIs[fishNumber-1])
            
            return AllVisibleFrames_Y_TF
            
# Analyze Correlations between Test and Stimulus Fish
def analyze_correlations(Test_folder, testNumber, Stim_folder, stimNumber, SocialFrames, corrLength, threshold):
    
    # Analyze Test
    trackingFile = Test_folder + r'/tracking' + str(testNumber) + '.npz'    
    data = np.load(trackingFile)
    tracking = data['tracking']
    
    fx = tracking[:,0] 
    fy = tracking[:,1]
    bx = tracking[:,2]
    by = tracking[:,3]
    ex = tracking[:,4]
    ey = tracking[:,5]
    area = tracking[:,6]
    ort_test = tracking[:,7]
    motion_test = tracking[:,8]
        
    # Analyze Stim
    trackingFile = Stim_folder + r'/tracking' + str(stimNumber) + '.npz'    
    data = np.load(trackingFile)
    tracking = data['tracking']
    
    fx = tracking[:,0] 
    fy = tracking[:,1]
    bx = tracking[:,2]
    by = tracking[:,3]
    ex = tracking[:,4]
    ey = tracking[:,5]
    area = tracking[:,6]
    ort_stim = tracking[:,7]
    motion_stim = tracking[:,8]
    
    # Filter Tracking based on Social Frames
    motion_test = motion_test[SocialFrames]
    motion_stim = motion_stim[SocialFrames]
    goodTracking = np.where((motion_test >= 0.0) * (motion_stim >= 0.0))
    goodTracking = goodTracking[0]
    motion_test = motion_test[goodTracking]
    motion_stim = motion_stim[goodTracking]
    
    # Threshold Motion Data to Remove Noise Correlations        
    baseline = np.median(motion_test)
    motion_test = motion_test - baseline
    sigma = np.std(motion_test)
    threshold_test = sigma*threshold
    motion_test = (motion_test > threshold_test) * motion_test

    baseline = np.median(motion_stim)
    motion_stim = motion_stim - baseline
    sigma = np.std(motion_stim)
    threshold_stim = sigma*threshold
    motion_stim = (motion_stim > threshold_stim) * motion_stim
    
    # Prepare "Motion" Arrays for Correlation Analysis (Pad with median "motion" value)
    padValue = np.median(motion_test)
    padding = np.zeros(corrLength)+padValue
    motion_test_padded = np.concatenate((padding, motion_test, padding), axis = 0)
    motion_stim_padded = np.concatenate((padding, motion_stim, padding), axis = 0)
    motion_test_padded_rev = motion_test_padded[::-1] # Scramble/Reverse Test
 
    # Compute Auto-Correlations
    auto_corr_test = np.correlate(motion_test_padded, motion_test, mode="valid")
    auto_corr_stim = np.correlate(motion_stim_padded, motion_stim, mode="valid")
    
    # Compute Cross-Correlations
    cross_corr = np.correlate(motion_test_padded, motion_stim, mode="valid")
    cross_corr_rev = np.correlate(motion_test_padded_rev, motion_stim, mode="valid")
    
    # Make Correlation Data Structure (2D array)
    corr_data = np.vstack((auto_corr_test, auto_corr_stim, cross_corr, cross_corr_rev))
    
    return corr_data

# Analyze Bouts of Test and Stimulus Fish
def analyze_bouts(testFolder, testNumber, stimFolder, stimNumber, visibleFrames, btaLength, threshold, testROI, stimROI):
    
    # Analyze Test
    trackingFile = testFolder + r'/tracking' + str(testNumber) + '.npz'    
    data = np.load(trackingFile)
    tracking = data['tracking']
    
    fx_test = tracking[:,0] 
    fy_test = tracking[:,1]
    bx_test = tracking[:,2]
    by_test = tracking[:,3]
    ex_test = tracking[:,4]
    ey_test = tracking[:,5]
    area_test = tracking[:,6]
    ort_test = tracking[:,7]
    motion_test = tracking[:,8]
        
    # Analyze Stim
    trackingFile = stimFolder + r'/tracking' + str(stimNumber) + '.npz'    
    data = np.load(trackingFile)
    tracking = data['tracking']
    
    fx_stim = tracking[:,0] 
    fy_stim = tracking[:,1]
    bx_stim = tracking[:,2]
    by_stim = tracking[:,3]
    ex_stim = tracking[:,4]
    ey_stim = tracking[:,5]
    area_stim = tracking[:,6]
    ort_stim = tracking[:,7]
    motion_stim = tracking[:,8]
    
    # Filter Tracking based on Social Frames (set to 0) - interpolate?
    motion_test[motion_test < 0.0] = 0.0
    motion_stim[motion_stim < 0.0] = 0.0
    
    # Compute Signal for bout detection (smoothed motion signal)   
    bout_filter = np.array([0.25, 0.25, 0.25, 0.25])
    boutSignal_test = signal.fftconvolve(motion_test, bout_filter, 'same')    
    boutSignal_stim = signal.fftconvolve(motion_stim, bout_filter, 'same')

    # Determine Threshold levels
    # - Determine the largest 100 values and take the median
    # - Use 10% of max level, divide by 10, for the base threshold (sigma)
    sorted_motion = np.sort(motion_test)
    max_norm = np.median(sorted_motion[-100:])    
    sigma_test = max_norm/10
    threshold_test = sigma_test*threshold    
    # - - - -
    print (threshold_test, max_norm)
    # - - - -
    sorted_motion = np.sort(motion_stim)
    max_norm = np.median(sorted_motion[-100:])    
    sigma_stim = max_norm/10
    threshold_stim = sigma_stim*threshold
    
    # Extract Bouts from Tracking Data
    bouts_test = SCZU.extract_bouts_from_motion(bx_test, by_test, ort_test, boutSignal_test, threshold_test, threshold_test-sigma_test, testROI, True)
    bouts_stim = SCZU.extract_bouts_from_motion(bx_stim, by_stim, ort_stim, boutSignal_stim, threshold_stim, threshold_stim-sigma_stim, stimROI, False)
    
    # Get Info about each bout (Align on STARTS!!)
    peaks_test = bouts_test[:, 1]
    peaks_stim = bouts_stim[:, 1]
    peaks_test = peaks_test.astype(int)
    peaks_stim = peaks_stim.astype(int)
    
    # Position at bout onset, offset

    # On Social(Visible side) at bout onset
    visible_testDuringStim = visibleFrames[peaks_stim]
    visible_stimDuringTest = visibleFrames[peaks_test]

    # Orientation of other fish during bout Peak
    # Zero degrees is towards bout generating fish, 180 is away, 90 is facing with REye, -90 is LEye
    # Seperate Orientations based on Chamber (1-6)
    # 1 - test right
    # 2 - test left
    # 3 - test right
    # 4 - test left
    # 5 - test right
    # 6 - test left

    ortOfTestDuringStim = ort_test[peaks_stim]
    ortOfStimDuringTest = ort_stim[peaks_test]
    
    # Adjust orientations so 0 is always pointing towards "other" fish
    if testNumber%2 == 0: # Test Fish facing Left
        for i,ort in enumerate(ortOfTestDuringStim):
            if ort >= 0: 
                ortOfTestDuringStim[i] = ort - 180
            else:
                ortOfTestDuringStim[i] = ort + 180    
    if stimNumber%2 == 1: # Stim fish facing Left
        for i,ort in enumerate(ortOfStimDuringTest):
            if ort >= 0: 
                ortOfStimDuringTest[i] = ort - 180
            else:
                ortOfStimDuringTest[i] = ort + 180  

    # Concatenate into Bouts Structure    
    bouts_test = np.hstack((bouts_test, np.transpose(np.atleast_2d(visible_stimDuringTest))))
    bouts_stim = np.hstack((bouts_stim, np.transpose(np.atleast_2d(visible_testDuringStim))))    
    
    bouts_test = np.hstack((bouts_test, np.transpose(np.atleast_2d(ortOfStimDuringTest))))
    bouts_stim = np.hstack((bouts_stim, np.transpose(np.atleast_2d(ortOfTestDuringStim))))

    return bouts_test, bouts_stim

# Compute BTA of Test and Stimulus Fish for different measures
def compute_BTA(bouts_test, bouts_stim, output_test, output_stim, btaLength):
        
    # Get Info about each bout (Align on Peaks!!)
    peaks_test = bouts_test[:, 1]
    peaks_stim = bouts_stim[:, 1]
    peaks_test = peaks_test.astype(int)
    peaks_stim = peaks_stim.astype(int)

    # Allocate Space for BTAs    
    BTA_test = np.zeros((np.size(peaks_test),2*btaLength,2))
    BTA_stim = np.zeros((np.size(peaks_stim),2*btaLength,2))

    # Pad OUTPUT variable for alignment
    padValue = np.median(output_test)
    padding = np.zeros(btaLength)+padValue
    output_test_padded = np.concatenate((padding, output_test, padding), axis = 0)
    padValue = np.median(output_stim)
    padding = np.zeros(btaLength)+padValue
    output_stim_padded = np.concatenate((padding, output_stim, padding), axis = 0)

    # Compute Burst Triggered Average (Auto-Corr)
    BTA_test[:,:,0] = SCZU.burst_triggered_alignment(peaks_test, output_test_padded, 0, btaLength*2)
    BTA_stim[:,:,0] = SCZU.burst_triggered_alignment(peaks_stim, output_stim_padded, 0, btaLength*2)

    # Compute Burst Triggered Average
    BTA_test[:,:,1] = SCZU.burst_triggered_alignment(peaks_test, output_stim_padded, 0, btaLength*2)
    BTA_stim[:,:,1] = SCZU.burst_triggered_alignment(peaks_stim, output_test_padded, 0, btaLength*2)
    
    return BTA_test, BTA_stim


#%% New figures 230920       
def plotBarWithSEM(variable, conditionNames, NS_data, S_data, linewidth=1.4, back_color=[0.5,0.5,0.5,1],bar_colorsS=(None, None, None), fignames=['NS', 'S'], att='', ylim=None):
    # Create a figure for the NS and S data
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Loop through NS and S data
    for idx, data, title, bar_colors in zip([0, 1], [NS_data, S_data], ['NS', 'S'], bar_colorsS):
        # Create a list to store means and SEMs for each condition
        means = []
        sems = []

        for i, name in enumerate(conditionNames):
            condition_data = data[i]
            condition_data = [float(val) for val in condition_data if val is not None and str(val).replace(".", "").replace("-", "").isdigit()]

            mean = np.mean(condition_data)
            sem = stats.sem(condition_data, nan_policy='omit')

            means.append(mean)
            sems.append(sem)

        # Manually set the color of the first bar to black, and keep the rest as default
        if bar_colors == None:
            bar_colors = ["black"] + ["lightgrey"] * (len(conditionNames) - 1)

        # Plot the bar chart with SEM error bars
        x = np.arange(len(conditionNames))
        axes[idx].bar(x, means, yerr=sems, capsize=5, color=bar_colors, alpha=0.8, linewidth=0)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(conditionNames, rotation=45, ha='right')
        axes[idx].set_ylabel(variable, fontsize=16, fontweight='bold')
        axes[idx].tick_params(axis='both', which='major', width=1.5)
        # Remove the right and top spines
        axes[idx].spines['right'].set_visible(False)
        axes[idx].spines['top'].set_visible(False)
        axes[idx].spines['left'].set_linewidth(1.5)
        axes[idx].spines['bottom'].set_linewidth(1.5)
        axes[idx].set_title(fignames[idx] + variable + '_' + att, fontsize=16, fontweight='bold')

        if ylim is not None:
            axes[idx].set_ylim(ylim)

        # Add horizontal dotted line and shaded area for the first data point
        axes[idx].hlines(means[0], -0.5, len(conditionNames) - 0.5, colors='black', linestyles='dotted', linewidth=linewidth/1.5)
        # axes[idx].fill_between([-0.5, len(conditionNames) - 0.5], means[0] - sems[0], means[0] + sems[0], color=back_color, alpha=0.3, lw=0)

    # Create a figure for the difference between S and NS data
    diff_fig, diff_axes = plt.subplots(figsize=(6, 6))
    diff_data = [np.array(s) - np.array(ns) for s, ns in zip(S_data, NS_data)]
    diff_means = [np.mean(data) for data in diff_data]
    diff_sems = [stats.sem(data, nan_policy='omit') for data in diff_data]

    # Use bar_colorsS[2] for the colors of the difference bars
    diff_colors = bar_colorsS[2]

    # Plot the bar chart with SEM error bars
    x = np.arange(len(conditionNames))
    diff_axes.bar(x, diff_means, yerr=diff_sems, capsize=5, color=diff_colors, alpha=0.8, linewidth=0)
    diff_axes.set_xticks(x)
    diff_axes.set_xticklabels(conditionNames, rotation=45, ha='right')
    diff_axes.set_ylabel(f'{fignames[1]} - {fignames[0]} {variable}', fontsize=16, fontweight='bold')
    diff_axes.tick_params(axis='both', which='major', width=1.5)
    # Remove the right and top spines
    diff_axes.spines['right'].set_visible(False)
    diff_axes.spines['top'].set_visible(False)
    diff_axes.spines['left'].set_linewidth(1.5)
    diff_axes.spines['bottom'].set_linewidth(1.5)
    diff_axes.set_title(f'Difference between {fignames[1]} and {fignames[0]} {variable}_' + att, fontsize=16, fontweight='bold')
    
    ylim=None
    if ylim is not None:
        diff_axes.set_ylim(ylim)

    # Add horizontal dotted line and shaded area for the first data point in the difference plot
    diff_axes.hlines(diff_means[0], -0.5, len(conditionNames) - 0.5, colors='black', linestyles='dotted', linewidth=linewidth)
    diff_axes.fill_between([-0.5, len(conditionNames) - 0.5], diff_means[0] - diff_sems[0], diff_means[0] + diff_sems[0], color=back_color, alpha=0.3, lw=0)

    plt.tight_layout()
    plt.show()

def plotBarWithSEM_orig(variable, conditionNames, NS_data, S_data, linewidth=2,bar_colorsS=(None,None),fignames=['NS', 'S'], att='', ylim=None):
    for idx, data, title, bar_colors in zip([0, 1], [NS_data, S_data], ['NS', 'S'], bar_colorsS):
        plt.figure()
        sns.set_palette("deep")

        # Create a list to store means and SEMs for each condition
        means = []
        sems = []

        for i, name in enumerate(conditionNames):
            condition_data = data[i]
            condition_data = [float(val) for val in condition_data if val is not None and str(val).replace(".", "").replace("-", "").isdigit()]

            mean = np.mean(condition_data)
            sem = stats.sem(condition_data, nan_policy='omit')

            means.append(mean)
            sems.append(sem)
            

        # Manually set the color of the first bar to black, and keep the rest as default
        if bar_colors==None:
            bar_colors = ["black"] + ["lightgrey"] * (len(conditionNames) - 1)
        # Plot the bar chart with SEM error bars
        x = range(len(conditionNames))
        plt.bar(x, means, yerr=sems, capsize=5, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.4)
        plt.xticks(x, conditionNames, rotation=45, ha='right')
        
        # plt.xlabel('Conditions', fontsize=12, fontweight='bold')
        plt.ylabel(variable, fontsize=16, fontweight='bold')
        plt.xticks(fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        ax = plt.gca()
        
        ax.tick_params(axis='both', which='major', width=1.5)
        # Remove the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)  # Set the right spine thickness to 2
        ax.spines['bottom'].set_linewidth(1.5)
        
        plt.title(fignames[idx] + variable + '_' + att, fontsize=16, fontweight='bold')

        if ylim is not None:
            plt.ylim(ylim)
        
        plt.tight_layout()


def plotSwarm(variable, conditionNames, NS_data, S_data, linewidth=2,back_color='lightgrey',swarm=True,se=False, swarm_colorsS=(None, None), fignames=['NS', 'S'], att='', ylim=None):
    for idx, data, title, swarm_colors in zip([0, 1], [NS_data, S_data], ['NS', 'S'], swarm_colorsS):
        plt.figure()

        # Create a list to store swarm positions and labels
        swarm_positions = []
        swarm_labels = []
        x_ticks = np.arange(len(conditionNames))
        for i, name in enumerate(conditionNames):
            condition_data = data[i]
            condition_data = [float(val) for val in condition_data if val is not None and str(val).replace(".", "").replace("-", "").isdigit()]

            # Calculate the x-positions for each condition
            x_position = i

            # Add jittered positions for each data point within the same condition
            jitter = np.random.normal(scale=0.05, size=len(condition_data))
            positions = np.full(len(condition_data), x_position) + jitter
            swarm_positions.extend(positions)
            swarm_labels.extend([name] * len(condition_data))

            # Manually set the color of the swarm points to grey
            swarm_color = "grey"

            # Plot the swarm points
            if swarm:
                plt.scatter(positions, condition_data, color=swarm_color, s=5, alpha=0.7)

            # Calculate and plot median and SEM lines with swarm_colors
            median = np.median(condition_data)
            if se:
                sem = np.std(condition_data) / np.sqrt(len(condition_data))
            else:
                sem = np.std(condition_data)

            # Use swarm_colors for the lines
            line_color = swarm_colors[i] if swarm_colors and len(swarm_colors) > i else 'black'

            plt.plot([x_position-0.2, x_position+0.2], [median, median], color=line_color, linewidth=linewidth)  # Median line
            plt.plot([x_position-0.1, x_position+0.1], [median - sem, median - sem], color=line_color, linewidth=linewidth)  # SEM lines
            plt.plot([x_position-0.1, x_position+0.1], [median + sem, median + sem], color=line_color, linewidth=linewidth)  # SEM lines
            plt.plot([x_position, x_position], [median + sem, median - sem], color=line_color, linewidth=linewidth)  # SEM lines
            
            # Add horizontal dotted line and shade the area for the first data point across the entire plot
            if i == 0:
                plt.hlines(median, x_ticks[0] - 0.2, x_ticks[-1] + 0.2, colors='black', linestyles='dotted', linewidth=2)
                plt.fill_between([x_ticks[0] - 0.2, x_ticks[-1] + 0.2], median - sem, median + sem, color=back_color, alpha=0.3,lw=0)

        # Set the x-axis ticks and labels to match the conditionNames
        plt.xticks(x_ticks, conditionNames, rotation=45, fontsize=12, fontweight='bold')

        plt.xlabel('Conditions', fontsize=16, fontweight='bold')
        plt.ylabel(variable, fontsize=16, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')
        ax = plt.gca()

        # Customize the axis spines thickness
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # Customize the tick linewidth for both x and y axes
        ax.tick_params(axis='both', which='major', width=1.5)

        # Remove the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.title(fignames[idx] + variable + '_' + att, fontsize=16, fontweight='bold')

        if ylim is not None:
            plt.ylim(ylim)

        plt.tight_layout()
    return ax
        
def plotDifferenceSwarm(variable, conditionNames, NS_data, S_data,linewidth=2,back_color='lightgrey', swarm=True, ci=False, swarm_colors=None, fignames=['Difference'], att='', ylim=None):
    plt.figure()

    # Create a list to store swarm positions and labels
    swarm_positions = []
    swarm_labels = []

    for i, condition_name in enumerate(conditionNames):
        ns_condition_data = NS_data[i]
        s_condition_data = S_data[i]

        ns_condition_data = [float(val) for val in ns_condition_data if val is not None and str(val).replace(".", "").replace("-", "").isdigit()]
        s_condition_data = [float(val) for val in s_condition_data if val is not None and str(val).replace(".", "").replace("-", "").isdigit()]

        # Calculate the difference between NS_data and S_data for each condition
        condition_data_diff = np.array(ns_condition_data) - np.array(s_condition_data)

        # Calculate the x-positions for each condition
        x_position = i

        # Add jittered positions for each data point within the same condition
        jitter = np.random.normal(scale=0.05, size=len(condition_data_diff))
        positions = np.full(len(condition_data_diff), x_position) + jitter
        swarm_positions.extend(positions)
        swarm_labels.extend([condition_name] * len(condition_data_diff))

        # Manually set the color of the swarm points to grey
        swarm_color = "grey"

        # Plot the swarm points
        if swarm:
            plt.scatter(positions, condition_data_diff, color=swarm_color, s=5, alpha=0.7)

        # Calculate and plot mean and CI lines with swarm_colors
        if ci:
            mean = np.median(condition_data_diff)
            ci_values = np.percentile(condition_data_diff, [25, 75])
            ci_lower = mean - ci_values[0]
            ci_upper = ci_values[1] - mean
        else:
            mean = np.median(condition_data_diff)
            sem = np.std(condition_data_diff) / np.sqrt(len(condition_data_diff))
            ci_lower = ci_upper = sem

        # Use swarm_colors for the lines
        line_color = swarm_colors[i] if swarm_colors and len(swarm_colors) > i else 'black'

        plt.plot([x_position-0.2, x_position+0.2], [mean, mean], color=line_color, linewidth=linewidth)  # Mean line
        plt.plot([x_position-0.1, x_position+0.1], [mean - ci_lower, mean - ci_lower], color=line_color, linewidth=linewidth)  # CI lines
        plt.plot([x_position-0.1, x_position+0.1], [mean + ci_upper, mean + ci_upper], color=line_color, linewidth=linewidth)  # CI lines
        plt.plot([x_position, x_position], [mean + ci_upper, mean - ci_lower], color=line_color, linewidth=linewidth)  # CI lines

    # Calculate and plot horizontal dotted lines and shaded area for the first condition
    first_condition_data = np.array(NS_data[0]) - np.array(S_data[0])
    if ci:
        first_mean = np.median(first_condition_data)
        first_ci_values = np.percentile(first_condition_data, [25, 75])
        first_ci_lower = first_mean - first_ci_values[0]
        first_ci_upper = first_ci_values[1] - first_mean
    else:
        first_mean = np.median(first_condition_data)
        first_sem = np.std(first_condition_data) / np.sqrt(len(first_condition_data))
        first_ci_lower = first_ci_upper = first_sem

    plt.axhline(y=first_mean, color='black', linestyle='--', linewidth=1, alpha=0.6)  # Dotted line at mean
    plt.fill_between([min(swarm_positions) - 0.5, max(swarm_positions) + 0.5],
                     first_mean - first_ci_lower, first_mean + first_ci_upper, color=back_color, alpha=0.2,lw=0)  # Shaded area

    # Set the x-axis ticks and labels to match the conditionNames
    x_ticks = np.arange(len(conditionNames))
    plt.xticks(x_ticks, conditionNames, rotation=45, fontsize=12, fontweight='bold',horizontalalignment='right')

    plt.xlabel('Conditions', fontsize=16, fontweight='bold')
    plt.ylabel(variable + ' Difference (NS - S)', fontsize=16, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    ax = plt.gca()

    # Customize the axis spines thickness
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Customize the tick linewidth for both x and y axes
    ax.tick_params(axis='both', which='major', width=1.5)

    # Remove the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.title(fignames[0] + variable + ' Difference' + '_' + att, fontsize=16, fontweight='bold')

    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    
def plotDifferenceSwarm1(variable, conditionNames, NS_data, S_data,linewidth=2, swarm=True, se=False, swarm_colors=None, fignames=['Difference'], att='', ylim=None):
    plt.figure()

    # Create a list to store swarm positions and labels
    swarm_positions = []
    swarm_labels = []

    for i, condition_name in enumerate(conditionNames):
        ns_condition_data = NS_data[i]
        s_condition_data = S_data[i]

        ns_condition_data = [float(val) for val in ns_condition_data if val is not None and str(val).replace(".", "").replace("-", "").isdigit()]
        s_condition_data = [float(val) for val in s_condition_data if val is not None and str(val).replace(".", "").replace("-", "").isdigit()]

        # Calculate the difference between NS_data and S_data for each condition
        condition_data_diff = np.array(ns_condition_data) - np.array(s_condition_data)

        # Calculate the x-positions for each condition
        x_position = i

        # Add jittered positions for each data point within the same condition
        jitter = np.random.normal(scale=0.05, size=len(condition_data_diff))
        positions = np.full(len(condition_data_diff), x_position) + jitter
        swarm_positions.extend(positions)
        swarm_labels.extend([condition_name] * len(condition_data_diff))

        # Manually set the color of the swarm points to grey
        swarm_color = "grey"

        # Plot the swarm points
        if swarm:
            plt.scatter(positions, condition_data_diff, color=swarm_color, s=5, alpha=0.7)
        # else:
            # plt.scatter(positions, condition_data_diff, color=swarm_color, s=5, alpha=0)
        # Calculate and plot median and SEM lines with swarm_colors
        median = np.median(condition_data_diff)
        if se:
            sem = np.std(condition_data_diff) / np.sqrt(len(condition_data_diff))
        else:
            sem = np.std(condition_data_diff)

        # Use swarm_colors for the lines
        line_color = swarm_colors[i] if swarm_colors and len(swarm_colors) > i else 'black'

        plt.plot([x_position-0.2, x_position+0.2], [median, median], color=line_color, linewidth=linewidth)  # Median line
        plt.plot([x_position-0.1, x_position+0.1], [median - sem, median - sem], color=line_color, linewidth=linewidth)  # SEM lines
        plt.plot([x_position-0.1, x_position+0.1], [median + sem, median + sem], color=line_color, linewidth=linewidth)  # SEM lines
        plt.plot([x_position, x_position], [median + sem, median - sem], color=line_color, linewidth=linewidth)  # SEM lines

    # Set the x-axis ticks and labels to match the conditionNames
    x_ticks = np.arange(len(conditionNames))
    plt.xticks(x_ticks, conditionNames, rotation=45, fontsize=12, fontweight='bold')

    plt.xlabel('Conditions', fontsize=16, fontweight='bold')
    plt.ylabel(variable + ' Difference (NS - S)', fontsize=16, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    ax = plt.gca()

    # Customize the axis spines thickness
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Customize the tick linewidth for both x and y axes
    ax.tick_params(axis='both', which='major', width=1.5)

    # Remove the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.title(fignames[0] + variable + ' Difference' + '_' + att, fontsize=16, fontweight='bold')

    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()





#ret,RTurns,LTurns,FSwims,BPS, allBouts, allBoutsDist, allBoutsOrt, boutAngles, LturnPC,boutStarts,boutEnds,_,_ = AZA.extractBouts(newFx,newFy,newOrt,newDistPerFrame, savepath=idF,plot=True)
# FIN
    