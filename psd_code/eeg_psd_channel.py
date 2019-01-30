import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from eeg_psd_csv import pick_len


def eeg_psd_pick_channel(name):
    c_file_name = 'psd_c_' + name + '.csv'
    p_file_name = 'psd_p_' + name + '.csv'
    c_df = pd.read_csv(c_file_name)
    p_df = pd.read_csv(p_file_name)
    del c_df['Unnamed: 0']
    del p_df['Unnamed: 0']
    c_df = c_df.T[2:pick_len+2]
    c_df = c_df.T
    name_df = c_df.columns
    c_df = c_df.mean()
    p_df = p_df.T[2:pick_len+2]
    p_df = p_df.T
    p_df = p_df.mean()
    subtraction=c_df-p_df
    i=0
    res=[]
    for sub in subtraction:
        if abs(sub)>0.03:
            res.append(name_df[i])
            # print(name_df[i])
        i+=1
    return res,name_df



def pick_channel():
    subBands = ['delta', 'theta', 'alpha1', 'alpha2', 'beta', 'gamma']
    dict={}
    picks=[]
    all_name=[]
    for band in subBands:
        res_channel,all_name = eeg_psd_pick_channel(band)
        for str in res_channel:
            if str in dict.keys():
                dict[str]+=1
            else:
                dict[str]=1
    for key in dict.keys():
        if dict[key]>0:
            picks.append(key)
    print(picks)
    print(len(picks))
    return picks,all_name
