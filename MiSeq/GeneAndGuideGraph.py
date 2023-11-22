# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:33:18 2023

@author: Tom
"""

# Create a graphical representation of a gene given, the start and end positions, the length of introns and exons, and the position of the target guides in the gene

#%% Gene Info
# gene_start, gene_end, intron_lengths, exon_lengths, start_ExonOrIntron = gene_info
grin2aa_gene_info = ['grin2aa',
                     27071749,
                     27277695, 
                     [237,2044,113497,18736,36248,16380,343,380,643,3383,2233,2036,946],
                     [83,337,408,593,115,203,172,154,126,230,161,188,239,6432],
                     'F']

grin2aa_guide_sites = [27188492,
                       27272022]

grin2ab_gene_info = ['grin2ab',8765624,8917902,[546,9209,96013,28820,3605,258,687,110,2806,2105,100,2190,414],[516,186,396,596,115,203,166,154,126,230,161,188,239,2140],'R']
grin2ab_guide_sites = [8810745,8768294] #C-8810518

gria3a_gene_info = ['gria3a',23356809,23429228,[218,11059,8812,5088,89,3386,233,1661,4011,3685,2556,15110,8218,5426],[286,159,240,188,54,162,168,105,108,207,374,199,248,115,255],'R']
gria3a_guide_sites = [23399344,23356991]

gria3b_gene_info = ['gria3b',12533560,12837432,[425,58461,74741,47197,1238,3125,12475,2129,27686,1886,13104,37428,16526,1183,301],[515,159,240,188,54,162,168,105,108,207,374,199,248,115,246,2878],'R']
gria3b_guide_sites = [12605867,12702791]

akap11_gene_info = ['akap11',17971935,18006632,[7283,107,3820,153,493,227,133,6468,2645,2567,2068],[110,97,111,54,132,241,4171,156,84,47,182,3349],'F']
akap11_guide_sites = [17987279,17984374,17988525]

cacna1g_gene_info = ['cacna1g',
                     27704015,
                     28033179,
                     [563,133937,19808,15942,14327,5705,3106,2946,3184,19001,601,89,1431,5060,5364,2020,1797,191,11385,17560,6255,20222,3441,198,2720,22648,4029],
                     [647,306,112,134,98,160,292,93,805,377,152,186,115,156,66,94,182,474,101,124,69,185,127,126,90,193,152,19],'F']
cacna1g_guide_sites = [27902791,27839519,27943201]

trioa_gene_info = ['trioa',41841147,41991104,[20419,6090,3439,13788,2363,1976,4160,3241,3105,1300,988,9541,3288,2194,105,2722,6303,73,1552,1226,139,1596,109,3099,112,617,1894,86,72,2339,3460,175,322,1650,1396,7303,2523,508,90,3407,81,666,99,1469,95,2978,2151,3034,1849,87,84,1767,300,2335,1652,2776],[157,75,115,193,543,123,192,132,231,123,192,170,175,196,167,120,192,150,115,116,123,195,116,67,110,70,90,93,112,191,102,140,100,244,296,171,96,149,67,104,70,90,93,51,78,192,181,728,248,142,28,163,132,69,201,139,1037],'R']
trioa_guide_sites = [41943864,41857496]

triob_gene_info = ['triob',8751815,8927425,[8670,7419,4802,219,14319,2559,1886,8901,2938,2830,683,7621,6030,8634,1378,1062,1878,77,4360,2679,148,369,2008,81,3003,595,8925,104,2682,2229,2456,329,374,21491,1120,1908,72,1497,341,76,2596,120,1294,2610,2899,85,1010,1915,3024,87,117,2287,96,105,2080,7575],[76,115,193,489,123,192,132,231,123,192,170,175,196,167,120,192,150,115,115,123,195,116,67,110,70,90,93,112,191,102,92,129,19,244,284,177,93,149,67,104,70,90,93,51,78,192,181,647,248,139,28,166,122,73,201,139,546],'R']
triob_guide_sites = [8771018,8810311]

xpo7_gene_info = ['xpo7',20070124, 20101182, [4697,159,78,984,153,104,3285,124,1181,970,571,76,999,1937,83,125,315,543,126,589,93,1444,99,1536,477,3615,1662],[37,147,94,167,66,105,166,74,120,147,173,194,106,136,74,81,64,109,107,89,108,83,215,139,161,99,128,1789],'E'] # 'E' indicates the first part of the gene is an exon 'I' is intron
xpo7_guide_sites = [20075435, 20083134, 20091595]

sp4_gene_info = ['sp4',2364552,2560616,[182775,2884,2834,2559,2605],[270,86,1408,193,200,251],'F']
sp4_guide_sites = [2547636,2551077,2550909]

nr3c2_gene_info = ['nr3c2',36935154,37087966,[3232,82131,31197,5515,5790,4299,14509,2600],[313,1783,128,117,315,145,131,158,450],'R']
nr3c2_guide_sites = [36963420,36938302,37082941]

herc1_gene_info = ['herc1',29189100,29271738,[89,454,2503,91,1718,1123,126,884,1251,379,71,82,3827,78,81,3170,274,1082,91,1848,2450,2104,101,3005,95,94,112,459,84,2091,82,84,1209,129,2237,1424,1188,88,2772,89,541,985,2024,89,1747,105,1636,80,316,1421,81,1577,123,113,655,84,2292,108,1522,249,1633,720,94,1551,172,116,93,120,900,108,508,1532,79,2500,82,2535],[944,96,195,312,97,144,128,145,172,135,166,126,213,153,133,192,216,174,115,121,175,283,121,52,269,219,121,238,170,156,107,111,171,240,84,530,758,196,161,117,208,116,209,269,53,201,175,268,214,263,128,137,201,131,212,121,147,102,174,318,66,126,139,182,156,123,103,170,139,170,170,166,77,253,153,306,627],'R']
herc1_guide_sites = [29222414,29189375,29227636]

hcn4_gene_info = ['hcn4',1004814,1143996,[60,43159,15,19387,10778,36537,1998,16833,7038,138,42],[366,242,414,25,162,219,147,241,165,471,21,725],'R']
hcn4_guide_sites = [1100056,1030351,1143803]

gene_info_list=[grin2aa_gene_info,
                grin2ab_gene_info,
                gria3a_gene_info,
                gria3b_gene_info,
                akap11_gene_info,
                cacna1g_gene_info,
                trioa_gene_info,
                triob_gene_info,
                xpo7_gene_info,
                sp4_gene_info,
                nr3c2_gene_info,
                herc1_gene_info,
                hcn4_gene_info,
                ]

guide_site_list=[grin2aa_guide_sites,
                 grin2ab_guide_sites,
                 gria3a_guide_sites,
                 gria3b_guide_sites,
                 akap11_guide_sites,
                 cacna1g_guide_sites,
                 trioa_guide_sites,
                 triob_guide_sites,
                 xpo7_guide_sites,
                 sp4_guide_sites,
                 nr3c2_guide_sites,
                 herc1_guide_sites,
                 hcn4_guide_sites,    
                 ]
#%% Function
import matplotlib.pyplot as plt

def plot_gene(gene_info, guide_sites):
    # Extract the gene type ('E' for exon, 'I' for intron)
    gene_name,gene_start, gene_end, intron_lengths, exon_lengths, gene_dir = gene_info
    
    # Check input data
    if len(intron_lengths)+1 != len(exon_lengths):
        print(f'intron and exon length lists are not the same size for gene {gene_name}, Check input data!')
        return
    for si,site in enumerate(guide_sites):
        if not gene_start <= site <= gene_end:
            print(f'Guide cut site {si} appears to be outside of this gene {gene_name}... check input data!')
            return
    # Create a figure and axis for plotting
    fig, ax = plt.subplots()

    # Track whether we are currently plotting introns or exons
    is_exon = True
    
    # Track whether we are on the reverse strand
    reverse = (gene_dir == 'R')
    
    if reverse:
        current_position = gene_end  
        intron_lengths = intron_lengths=[item*-1 for item in intron_lengths]
        exon_lengths = exon_lengths=[item*-1 for item in exon_lengths]
    else:
        current_position = gene_start
    
    intron_lengths.append(0)
    # Initialize the starting position
    for intron_length, exon_length in zip(intron_lengths, exon_lengths):
        if is_exon:
            ax.plot([current_position, current_position + exon_length], [1, 1], linewidth=9, color='blue')
            current_position += exon_length
        
            ax.plot([current_position, current_position + intron_length], [1, 1], linewidth=3, color='blue')
            current_position += intron_length
        else:
            
            ax.plot([current_position, current_position + intron_length], [1, 1], linewidth=3, color='blue')
            current_position += intron_length
        
            ax.plot([current_position, current_position + exon_length], [1, 1], linewidth=9, color='blue')
            current_position += exon_length
        
    # Plot CRISPR guide cut sites as arrows
    for si,site in enumerate(guide_sites):
        ax.arrow(site, 1.01, 0, -0.0075 , head_width=0, head_length=0, fc='red', ec='red')
            
        
    # Set the x-axis limits and labels
    # ax.set_xlim(0, total_gene_length)
    ax.set_xlabel('Genomic Position')
    
    # Remove y-axis and set y-ticks
    ax.set_yticks([])
    plt.ylim(0.96,1.02)

    # Set title
    ax.set_title(f'Gene Structure with CRISPR Guide Cut Sites - {gene_name}')

    # Display the plot
    plt.show()

#%% Main
if __name__ == "__main__":
    
    # Example gene information
    # gene_start, gene_end, intron_lengths, exon_lengths = gene_info
    # guide_sites are genomic positions without chromosome info
    
    for gene_info,guide_sites in zip(gene_info_list,guide_site_list):
        plot_gene(gene_info,guide_sites)