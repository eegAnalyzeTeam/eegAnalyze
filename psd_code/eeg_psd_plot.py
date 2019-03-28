import matplotlib.pyplot as plt
import numpy as np
import eeg_psd_channel
import pandas as pd

from eeg_psd_csv import pick_len



def eeg_psd_plot(name):
    c_file_name = 'psd_c_' + name + '.csv'
    p_file_name = 'psd_p_' + name + '.csv'
    c_df = pd.read_csv(c_file_name)
    p_df = pd.read_csv(p_file_name)
    del c_df['Unnamed: 0']
    del p_df['Unnamed: 0']

    c_df = c_df.T[2:pick_len+2]
    #c_df.plot(alpha=0.2)
    c_df = c_df.T
    name_df = c_df.columns
    # name_df=picks
    # c_df=c_df.get(picks)
    c_df = c_df.mean()
    p_df = p_df.T[2:pick_len+2]
    #p_df.plot(alpha=0.2)
    p_df = p_df.T
    # p_df = p_df.get(picks)
    p_df = p_df.mean()
    x = range(1, pick_len+1, 1)
    # c_df.plot(label = 'control')

    p_df.plot(label='patient')
    c_df.plot(label='control').set_ylabel('PSD')
    plt.title(name)
    plt.legend()
    plt.xticks(x, name_df[0:pick_len], rotation=60)
    plt.savefig(name + '.png')
    plt.close()


def plot_psd():
    subBands = ['delta', 'theta', 'alpha1', 'alpha2', 'beta', 'gamma']
    # picks = eeg_psd_channel.pick_channel()
    for band in subBands:
        eeg_psd_plot(band)
