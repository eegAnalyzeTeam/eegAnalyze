import mne
from mne import io
import matplotlib
raw=mne.io.read_raw_brainvision('jkdz_cc_03_20180430_close.vhdr',preload=True)#可更改eeg文件名
raw.set_montage(mne.channels.read_montage("standard_1020"))
raw.filter(0.5, 4, fir_design='firwin')#使用fir滤波器通带频率为0.5到4HZ，可以更改为其他通带的频率
#print(raw.info)#从info中可以看通带频率是否更改
raw.plot()#绘图

