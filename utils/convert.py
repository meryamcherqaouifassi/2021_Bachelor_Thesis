#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:03:43 2022

@author: mcherqao
"""

import pandas as pd
import numpy as np

data_path = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/data/data_casa'

dataTxt = data_path+'/Casablanca_2000_03.txt'

def txt_to_df(file_name, end, start):
    # table headers
    headers = ["PRES", "HGHT", "TEMP", "DWPT", "RELH", "MIXR", "DRCT", "SKNT", "THTA", "THTE", "THTV"]
    
    firstStartDetected = False
    
    lines = []
    
    with open(file_name, 'r') as f:
        lines = f.readlines()
        
    # remove first 9 items from lines
    lines[0:8] = []
    
    # array of index range to remove in lines
    indexesToRemove = []
    
    # current index in lines array
    index = 0
    
    for line in lines :
        index = index + 1
        if start in line :
            if firstStartDetected == False :
                firstStartDetected = True
            else :
                firstStartDetected = False
                indexesToRemove[len(indexesToRemove) - 1].append(index + 3)
                continue
        if end in line :
            indexesToRemove.append([index - 1])
            continue

    # total item removed
    removedCount = 0
    
    # delete non desired lines in lines
    for index in indexesToRemove :
        if len(index) == 2 :
            lines[index[0] - removedCount:index[1] - removedCount] = []
            removedCount = removedCount + (index[1] - index[0])
        elif len(index) == 1 :
            lines[index[0] - removedCount:] = []
    
 
    lines = [line.replace("       ", "      ?") for line in lines]
    lines = [line.split() for line in lines]
    
    
    df = pd.DataFrame(lines[1:], columns=headers)
    df = df.replace('?', np.nan)
   
    return df

start = "-----------------------------------------------------------------------------"
end = "</PRE><H3>Station information and sounding indices</H3><PRE>"     

dfFromTxtFile = txt_to_df(dataTxt, end, start);