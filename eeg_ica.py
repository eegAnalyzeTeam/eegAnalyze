import mne
import os.path as op
import numpy as np
from matplotlib import pyplot as plt

data_path='D:\health_control\eyeopen'
fname=data_path+'\jkdz_dhl_20180411_open.vhdr'  #根据实际情况改变路径
print(__doc__)
raw=mne.io.read_raw_brainvision(fname,preload=True)
raw.set_montage(mne.channels.read_montage("standard_1020"))
raw_tmp = raw.copy()
raw_tmp.filter(1, None, fir_design="firwin")
print(raw)

method=input("Which ICA method do you want to use?\n You can choose fastica,picard,infomax or extended-infomax\n")
ica = mne.preprocessing.ICA(method=method,random_state=1)
ica.fit(raw_tmp)
print(ica)
ica.plot_components(inst=raw_tmp)
ica.plot_sources(inst=raw_tmp)
raw.plot(n_channels=18,start=50,duration=30,scalings=dict(eeg=150e-6, eog=750e-6))
ica.exclude =[]   #根据判断结果添加需要排除分量的序号
raw_corrected = raw.copy()
ica.apply(raw_corrected)
raw_corrected.plot(n_channels=18,start=50,duration=30,scalings=dict(eeg=150e-6, eog=750e-6))
