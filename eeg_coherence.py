# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 12:52:50 2018

@author: 64942
"""

from scipy import signal
import mne 
import numpy as np
import csv
import eeg_sub_bands
import matplotlib.pyplot as plt
import matplotlib.mlab as plot


def coh_get_Section(f,a,b):
    start=-1
    end=-1
    for i in range(0,len(f)):
        if start ==-1 and f[i]>a:
            start=i
        if end == -1 and f[i]>b:
            end=i
            break
    return start,end
start=-1
end=-1

def coh_get_channel_names():
    raw = mne.io.read_raw_brainvision('/home/public2/eegData/health_control/eyeclose/jkdz_cc_20180430_close.vhdr',
                                      preload=True)
    channel_names = []
    for i in raw.info['ch_names']:
        if i != 'Oz':
            if i != 'ECG':
                channel_names.append(i)
    return channel_names[0:-1]

def coherence(x,y,a,b):
    fs=100
    f,Cxy=signal.coherence(x,y,fs,nperseg=20)
    global start
    global end
    if start ==-1:
        start,end = coh_get_Section(f,a,b)
    return(np.mean(Cxy[start:end]))

def eeg_coherence(raw):
    fileread=open('eeg_coherence.csv','w',newline='')
    writer=csv.writer(fileread)
    writer.writerow(coh_get_channel_names())
    for i in range(0,62):
        data=[]
        for j in range(0,62):
            print('channel: ', i, ' ', j)
            data+=[coherence(raw[61-i][0][0],raw[j][0][0],7.5,12.5)]
        writer.writerow(data)
    fileread.close

def coherence_ofList(raw_list,flag):
    print(raw_list)
    columns = coh_get_channel_names()
    if flag=='c':
        fileread=open('eeg_coherence_c.csv','w',newline='')
        writer=csv.writer(fileread)
        writer.writerow(columns)
        for i in range(0,62):
            data=[]
            for j in range(0,62):
                t=[]
                if (61-i)<=j:
                    tempread = open('eeg_coh_anova/'+str(columns[61-i])+'_'+str(columns[j])+'_c'+'.csv', 'w', newline='')
                    tempwriter = csv.writer(tempread)
                    tempwriter.writerow(['id','coherence'])
                for raw in raw_list:
                    t+=[coherence(raw[61-i][0][0],raw[j][0][0],7.5,12.5)]
                if (61-i)<=j:
                    for temp_coh in t:
                        tempwriter.writerow(['0',temp_coh])
                    tempread.close
                s=np.mean(t)
                data+=[s]
                print('c'+ str(i)+' '+str(j))
            writer.writerow(data)
        fileread.close
    else:
        fileread=open('eeg_coherence_p.csv','w',newline='')
        writer=csv.writer(fileread)
        writer.writerow(columns)
        for i in range(0,62):
            data=[]
            for j in range(0,62):
                t=[]
                if (61-i)<=j:
                    tempread = open('eeg_coh_anova/'+str(columns[61-i])+'_'+str(columns[j])+'_p'+'.csv', 'w', newline='')
                    tempwriter = csv.writer(tempread)
                    tempwriter.writerow(['id','coherence'])
                for raw in raw_list:
                    t+=[coherence(raw[61-i][0][0],raw[j][0][0],7.5,12.5)]
                if (61-i)<=j:
                    for temp_coh in t:
                        tempwriter.writerow(['1',temp_coh])
                    tempread.close
                s=np.mean(t)
                data+=[s]
                print('c'+ str(i)+' '+str(j))
            writer.writerow(data)
        fileread.close

def coh(control_raw,patient_raw):
    control_raw_temp=[]
    patient_raw_temp=[]
    count=0
    for (eid,raw) in control_raw.items():
        raw.load_data()
        raw.drop_channels(['Oz', 'ECG'])
        # raw = eeg_sub_bands.eeg_sub_bands(raw, 'alpha1')
        raw = raw.resample(100, npad='auto')
        print(count)
        count+=1
        control_raw_temp.append(raw)
    coherence_ofList(control_raw_temp,'c')
    count=0
    for (eid,raw) in patient_raw.items():
        raw.load_data()
        raw.drop_channels(['Oz', 'ECG'])
        # raw = eeg_sub_bands.eeg_sub_bands(raw, 'alpha1')
        raw = raw.resample(100, npad='auto')
        print(count)
        count+=1
        patient_raw_temp.append(raw)
    coherence_ofList(patient_raw_temp,'p')
        
            
                    
# raw_fname= 'jkdz_cc_01_20180430_close.vhdr'
# raw=mne.io.read_raw_brainvision(raw_fname, preload=True)
# raw.drop_channels(['Oz', 'ECG'])
# alpha1_raw = eeg_sub_bands.eeg_sub_bands(raw, 'alpha1')
# alpha1_raw = alpha1_raw.resample(100, npad='auto')
# eeg_coherence(alpha1_raw)
# print('end')
# x,y = alpha1_raw[63][0][0], alpha1_raw[0][0][0]
# f,Cxy=signal.coherence(x,y, 5000, nperseg=1024)
# plt.semilogy(f, Cxy)
# #plt.show()
# coh, f = plt.cohere(x,y,256,  1./.01)
# print(f)
# plt.show()
