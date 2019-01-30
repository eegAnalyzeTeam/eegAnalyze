import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import mne
import matplotlib.image as mg

def readfile():
    N = 62
    file = ['eeg_coherence_c.csv', 'eeg_coherence_p.csv']
    control_csv = pd.read_csv(file[0])
    columns=control_csv.columns.tolist()

    patient_csv=pd.read_csv(file[1])

    xlabels = columns
    ylabels = columns[::-1]
    return xlabels,ylabels,control_csv,patient_csv



def coherence_difference_plot():
    N=62
    xlabels, ylabels, control_csv, patient_csv=readfile()
    lists = []
    for i in range(N):
        lists.append(list(map(lambda x,y:abs(x-y),control_csv.loc[i][0:N],patient_csv.loc[i][0:N])))

    _list = np.array(lists)
    plt.style.use('classic')
    figure = plt.figure(facecolor='lightgrey')
    ax = figure.add_subplot(111)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    _map = ax.imshow(_list, interpolation='nearest', cmap=plt.cm.rainbow, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(mappable=_map)

    plt.title('difference')
    plt.savefig('coherence_d.png')
    plt.close()


def conherence_sort():
    N=62
    xlabels, ylabels, control_csv, patient_csv = readfile()


    lists = []
    for i in range(N):
        lists.append(list(map(lambda x, y: abs(x - y), control_csv.loc[i][0:N], patient_csv.loc[i][0:N])))
    difference =np.array(lists)

    dict={}
    for x in range(N):
        for y in range(N):
            if str(xlabels[x])+"_"+str(ylabels[y]) in dict.values() or   str(ylabels[y])+"_"+str(xlabels[x]) in dict.values():
                continue
            dict[xlabels[x]+"_"+ylabels[y]]=difference[x][y]

    res = sorted(dict.items(), key=lambda x: x[1],reverse=True)
    print(res)
    file = open('sort.txt', 'w')
    for x in res:
        file.writelines(str(x) + "\r\n")
    file.close()

def check_difference(x,y):
    if abs(x - y)>0.1:
        return 1
    else:
        return 0


def gray_plot():
    N = 62
    xlabels, ylabels, control_csv, patient_csv = readfile()
    lists = []
    for i in range(N):
        lists.append(list(map(check_difference, control_csv.loc[i][0:N], patient_csv.loc[i][0:N])))

    _list = np.array(lists)
    plt.style.use('classic')
    figure = plt.figure(facecolor='lightgrey')
    ax = figure.add_subplot(111)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    _map = ax.imshow(_list, interpolation='nearest', cmap=plt.cm.rainbow, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(mappable=_map)

    plt.title('difference')
    plt.savefig('coherence_gray.png')
    plt.close()


def brain_plot():
    N = 62
    xlabels, ylabels, control_csv, patient_csv = readfile()
    raw_fname = 'jkdz_cc_01_20180430_close.vhdr'
    raw=mne.io.read_raw_brainvision(raw_fname, preload=True)
    raw.drop_channels(['Oz', 'ECG'])
    epoch=mne.Epochs(raw, mne.find_events(raw))
    pos = mne.find_layout(epoch.info).pos
    print(raw.info)
    image,_ =mne.viz.plot_topomap(control_csv,pos)
    mne.viz.tight_layout()
    mg.imsave('test_control',image)


# conherence_sort()
# gray_plot()

brain_plot()

print('end')