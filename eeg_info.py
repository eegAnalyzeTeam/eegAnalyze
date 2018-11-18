import mne
import os
import numpy as np
from matplotlib import pyplot as plt
import csv

data_path='D:\health_control\eyeopen'  #根据实际情况改变路径
data_path1='D:\health_control\eyeclose'

dirs=os.listdir(data_path)
dirs1=os.listdir(data_path1)
filename_open=[]
filename_close=[]
for i in dirs:
    if os.path.splitext(i)[1]==".eeg":
        filename_open.append(os.path.splitext(i)[0])

dirs1=os.listdir(data_path1)
for i in dirs1:
    if os.path.splitext(i)[1]==".eeg":
        filename_close.append(os.path.splitext(i)[0])
        
fileread=open('eeg_info.csv','w',newline='')
writer=csv.writer(fileread)

writer.writerow(['id','n_channels*n_times','events'])
for i in filename_open:
    raw=mne.io.read_raw_brainvision(data_path+'\\'+i+'.vhdr')
    a=str(raw)
    data=[i,a[a.index(':')+1:a.index(')')+1],raw.info['events']]
    writer.writerow(data)
for i in filename_close:
    raw=mne.io.read_raw_brainvision(data_path1+'\\'+i+'.vhdr')
    a=str(raw)
    data=[i,a[a.index(':')+1:a.index(')')+1],raw.info['events']]
    writer.writerow(data)  
