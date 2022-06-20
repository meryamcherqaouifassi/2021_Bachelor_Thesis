# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:03:43 2022

@author: mcherqao
"""

# imports
import pandas as pd
import numpy as np
import os

# get data
data_path = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/data/data_casa'
def get_files(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))
    return files

# convert int month to string of date format
def string_to_month(month):
    if month == 'Jan':
        month = '01'
    elif month == 'Feb':
        month = '02'
    elif month == 'Mar':
        month = '03'
    elif month == 'Apr':
        month = '04'
    elif month == 'May':
        month = '05'
    elif month == 'Jun':
        month = '06'
    elif month == 'Jul':
        month = '07'
    elif month == 'Aug':
        month = '08'
    elif month == 'Sep':
        month = '09'
    elif month == 'Oct':
        month = '10'
    elif month == 'Nov':
        month = '11'
    elif month == 'Dec':
        month = '12'
    return month

# save text data in dataframe
def txt_to_df(file_name, end, start):
    # table headers
    headers = ["DATE", "PRES", "HGHT", "TEMP", "DWPT", "RELH", "MIXR", "DRCT", "SKNT", "THTA", "THTE", "THTV"]
    
    first_start_detected = False
    
    lines = []
    lines_with_date = []
    
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    # save first eight lines            
    firstEightLines = lines[0:8]
    
    # remove first 9 items from lines
    lines[0:8] = []
    
    # array of index range to remove in lines
    indexes_to_remove = []
    
    # current index in lines array
    index = 0
    
    # keep only file name and split it by '_' to get city, year and month
    txt_date = file_name

    actual_date = ""
    
    for line in firstEightLines :
        if line.startswith(start_date_line):
            splitted_date = line[-21:-6].split(" ")
            actual_date = splitted_date[1] + "-" + string_to_month(splitted_date[2]) + "-" + splitted_date[3] + "T" + splitted_date[0]
            actual_date = actual_date[:-1]
            print(actual_date)
    
    if actual_date == "":
        print("empty file: " + txt_date)
        return
        
    for line in lines :
        index = index + 1

        # if line start with start_date_line value keep only 16 last characters
        if line.startswith(start_date_line):
            splitted_date = line[-21:-6].split(" ")
            actual_date = splitted_date[1] + "-" + string_to_month(splitted_date[2]) + "-" + splitted_date[3] + "T" + splitted_date[0]
            actual_date = actual_date[:-1]
         
        # add edited line in a new array
        lines_with_date.append(actual_date + " " + line)

        if start in line :
            if first_start_detected == False :
                first_start_detected = True
            else :
                first_start_detected = False
                indexes_to_remove[len(indexes_to_remove) - 1].append(index + 3)
                continue
        if end in line :
            indexes_to_remove.append([index - 1])
            continue
        
    # total item removed
    removed_count = 0

    # delete non desired lines in lines
    for index in indexes_to_remove :
        if len(index) == 2 :
            lines_with_date[index[0] - removed_count:index[1] - removed_count] = []
            removed_count = removed_count + (index[1] - index[0])
        elif len(index) == 1 :
            lines_with_date[index[0] - removed_count:] = []
    
        
    lines_with_date = [line.replace("       ", "      ?") for line in lines_with_date]
    lines_with_date = [line.split() for line in lines_with_date]
    
    
    df = pd.DataFrame(lines_with_date[1:], columns=headers)
    df = df.replace('?', np.nan)
   
    return df

# start and end values
start = "-----------------------------------------------------------------------------"
end = "</PRE><H3>Station information and sounding indices</H3><PRE>" 
start_date_line = "<H2>"

# conversion
to_df_from_txt = []
files = get_files(data_path)
for file in files : 
    to_df_from_txt.append(txt_to_df(file, end, start))
    
def merge_dfs(dfs):
    return pd.concat(dfs, ignore_index=True)

# remove all items of type Nonetype Object from to_df_from_txt
to_df_from_txt = [df for df in to_df_from_txt if df is not None]

data_casa = merge_dfs(to_df_from_txt)
    
# export and save to csv file
#file_path = '/work/FAC/FGSE/IDYST/tbeucler/default/meryam/2021_Bachelor_Thesis/files'
#data_casa.to_csv(file_path+'/data_casa.csv', index = False)