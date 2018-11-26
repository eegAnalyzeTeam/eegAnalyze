import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def eeg_psd_plot(name):
    c_file_name = 'psd_c_' + name + '.csv'
    p_file_name = 'psd_p_' + name + '.csv'
    c_df=pd.read_csv(c_file_name)
    p_df=pd.read_csv(p_file_name)
    del c_df['Unnamed: 0']
    del p_df['Unnamed: 0']
    c_df=c_df.T[2:66]
    c_df=c_df.T
    name_df=c_df.columns
    c_df=c_df.mean()
    p_df=p_df.T[2:66]
    p_df=p_df.T
    p_df=p_df.mean()
    x=range(1,65,1)
    #c_df.plot(label = 'control')
    p_df.plot(label = 'patient')
    c_df.plot(label = 'control').set_ylabel('PSD')
    plt.title(name)
    plt.legend()
    plt.xticks(x, name_df[0:64], rotation=60)
    plt.savefig(name + '.jpg')
    plt.close()
        

def plot_psd():
    subBands = ['delta', 'theta', 'alpha1', 'alpha2', 'beta', 'gamma']
    for band in subBands:
        eeg_psd_plot(band)
