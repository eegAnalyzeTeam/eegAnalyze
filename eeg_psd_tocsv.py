import mne
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd

#eeg_psd函数，输入.vhdr文件路径，输出各频段的PSD
def eeg_psd(fname):
    #data_path='D:\health_control\eyeopen'
    #fname=data_path+'\jkdz_dhl_20180411_open.vhdr'
    print(__doc__)
    raw=mne.io.read_raw_brainvision(fname,preload=True)

    psds_delta,freqs1=mne.time_frequency.psd_welch(raw,fmin=0.5,fmax=4,n_fft=2048,n_jobs=1)

    psds_theta,freqs2=mne.time_frequency.psd_welch(raw,fmin=4,fmax=7,n_fft=2048,n_jobs=1)

    psds_alpha1,freqs3=mne.time_frequency.psd_welch(raw,fmin=8,fmax=10,n_fft=2048,n_jobs=1)

    psds_alpha2 ,freqs4=mne.time_frequency.psd_welch(raw,fmin=10,fmax=12,n_fft=2300,n_jobs=1)

    psds_beta,freqs5=mne.time_frequency.psd_welch(raw,fmin=13,fmax=30,n_fft=256,n_jobs=1)

    psds_gamma,freqs6=mne.time_frequency.psd_welch(raw,fmin=30,fmax=40,n_fft=512,n_jobs=1)
    
    
    psd_subfreq={}
    psd_subfreq['delta']=psds_delta
    psd_subfreq['theta']=psds_theta
    psd_subfreq['alpha1']=psds_alpha1
    psd_subfreq['alpha2']=psds_alpha2
    psd_subfreq['beta']=psds_beta
    psd_subfreq['gamma']=psds_gamma
    
    return psd_subfreq



data_path1='D:\eeg\health_control\eyeopen'   #文件夹路径，可根据需要更改
data_path2='D:\eeg\health_control\eyeclose'
data_path3='D:\eeg\mdd_patient\eyeopen'
data_path4='D:\eeg\mdd_patient\eyeclose'


dirs1=os.listdir(data_path1)   #提取文件夹中所有的文件名
dirs2=os.listdir(data_path2)
dirs3=os.listdir(data_path3)
dirs4=os.listdir(data_path4)
filename_control_open=[]       #储存id的列表
filename_control_close=[]
filename_patient_open=[]
filename_patient_close=[]

#从所有的文件名中分离出id
for i in dirs1:
    if os.path.splitext(i)[1]==".eeg":
        filename_control_open.append(os.path.splitext(i)[0])


for i in dirs2:
    if os.path.splitext(i)[1]==".eeg":
        filename_control_close.append(os.path.splitext(i)[0])
        
for i in dirs3:
    if os.path.splitext(i)[1]==".eeg":
        filename_patient_open.append(os.path.splitext(i)[0])


for i in dirs4:
    if os.path.splitext(i)[1]==".eeg":
        filename_patient_close.append(os.path.splitext(i)[0])        
        
#建立以channel1~64为键，空列表为值的12个 control_delta ~  patient_gamma  字典  
chs=[]
for value in range(1,65):
    chs.append('channel'+str(value))
      
frame_control_delta={}
for j in chs:
    frame_control_delta[j]=[]
frame_control_theta={}
for j in chs:
    frame_control_theta[j]=[]
frame_control_alpha1={}
for j in chs:
    frame_control_alpha1[j]=[]
frame_control_alpha2={}
for j in chs:
    frame_control_alpha2[j]=[]
frame_control_beta={}
for j in chs:
    frame_control_beta[j]=[]
frame_control_gamma={}
for j in chs:
    frame_control_gamma[j]=[] 
    
frame_patient_delta={}
for j in chs:
    frame_patient_delta[j]=[]
frame_patient_theta={}
for j in chs:
    frame_patient_theta[j]=[]
frame_patient_alpha1={}
for j in chs:
    frame_patient_alpha1[j]=[]
frame_patient_alpha2={}
for j in chs:
    frame_patient_alpha2[j]=[]
frame_patient_beta={}
for j in chs:
    frame_patient_beta[j]=[]
frame_patient_gamma={}
for j in chs:
    frame_patient_gamma[j]=[]      
        
        
        
#逐个读取eeg文件，运行eeg_psd并将结果储存在对应字典中        
for i in filename_control_open:
    fname=data_path1+'\\'+i+'.vhdr'
    psd_subfreq=eeg_psd(fname)
    for j in range(1,65):
        frame_control_delta['channel'+str(j)].append(psd_subfreq['delta'][j-1][0])
        frame_control_theta['channel'+str(j)].append(psd_subfreq['theta'][j-1][0])
        frame_control_alpha1['channel'+str(j)].append(psd_subfreq['alpha1'][j-1][0])
        frame_control_alpha2['channel'+str(j)].append(psd_subfreq['alpha2'][j-1][0])
        frame_control_beta['channel'+str(j)].append(psd_subfreq['beta'][j-1][0])
        frame_control_gamma['channel'+str(j)].append(psd_subfreq['gamma'][j-1][0])
for i in filename_control_close:
    fname=data_path2+'\\'+i+'.vhdr'
    psd_subfreq=eeg_psd(fname)
    for j in range(1,65):
        frame_control_delta['channel'+str(j)].append(psd_subfreq['delta'][j-1][0])
        frame_control_theta['channel'+str(j)].append(psd_subfreq['theta'][j-1][0])
        frame_control_alpha1['channel'+str(j)].append(psd_subfreq['alpha1'][j-1][0])
        frame_control_alpha2['channel'+str(j)].append(psd_subfreq['alpha2'][j-1][0])
        frame_control_beta['channel'+str(j)].append(psd_subfreq['beta'][j-1][0])
        frame_control_gamma['channel'+str(j)].append(psd_subfreq['gamma'][j-1][0])
 
 #将字典转换为data frame
df_control_delta=pd.DataFrame(frame_control_delta,index=filename_control_open+filename_control_close)
df_control_theta=pd.DataFrame(frame_control_theta,index=filename_control_open+filename_control_close)
df_control_alpha1=pd.DataFrame(frame_control_alpha1,index=filename_control_open+filename_control_close)
df_control_alpha2=pd.DataFrame(frame_control_alpha2,index=filename_control_open+filename_control_close)
df_control_beta=pd.DataFrame(frame_control_beta,index=filename_control_open+filename_control_close)
df_control_gamma=pd.DataFrame(frame_control_gamma,index=filename_control_open+filename_control_close)

#将data frame储存在对应.csv文件中（.csv文件将自动生成在该py文件所在目录下）
df_control_delta.to_csv('control_delta.csv',index=True)
df_control_theta.to_csv('control_theta.csv',index=True)
df_control_alpha1.to_csv('control_alpha1.csv',index=True)
df_control_alpha2.to_csv('control_alpha2.csv',index=True)
df_control_beta.to_csv('control_beta.csv',index=True)
df_control_gamma.to_csv('control_gamma.csv',index=True)

    
for i in filename_patient_open:
    fname=data_path3+'\\'+i+'.vhdr'
    psd_subfreq=eeg_psd(fname)
    for j in range(1,65):
        frame_patient_delta['channel'+str(j)].append(psd_subfreq['delta'][j-1][0])
        frame_patient_theta['channel'+str(j)].append(psd_subfreq['theta'][j-1][0])
        frame_patient_alpha1['channel'+str(j)].append(psd_subfreq['alpha1'][j-1][0])
        frame_patient_alpha2['channel'+str(j)].append(psd_subfreq['alpha2'][j-1][0])
        frame_patient_beta['channel'+str(j)].append(psd_subfreq['beta'][j-1][0])
        frame_patient_gamma['channel'+str(j)].append(psd_subfreq['gamma'][j-1][0])
for i in filename_patient_close:
    fname=data_path4+'\\'+i+'.vhdr'
    psd_subfreq=eeg_psd(fname)
    for j in range(1,65):
        frame_patient_delta['channel'+str(j)].append(psd_subfreq['delta'][j-1][0])
        frame_patient_theta['channel'+str(j)].append(psd_subfreq['theta'][j-1][0])
        frame_patient_alpha1['channel'+str(j)].append(psd_subfreq['alpha1'][j-1][0])
        frame_patient_alpha2['channel'+str(j)].append(psd_subfreq['alpha2'][j-1][0])
        frame_patient_beta['channel'+str(j)].append(psd_subfreq['beta'][j-1][0])
        frame_patient_gamma['channel'+str(j)].append(psd_subfreq['gamma'][j-1][0])

df_patient_delta=pd.DataFrame(frame_patient_delta,index=filename_patient_open+filename_patient_close)
df_patient_theta=pd.DataFrame(frame_patient_theta,index=filename_patient_open+filename_patient_close)
df_patient_alpha1=pd.DataFrame(frame_patient_alpha1,index=filename_patient_open+filename_patient_close)
df_patient_alpha2=pd.DataFrame(frame_patient_alpha2,index=filename_patient_open+filename_patient_close)
df_patient_beta=pd.DataFrame(frame_patient_beta,index=filename_patient_open+filename_patient_close)
df_patient_gamma=pd.DataFrame(frame_patient_gamma,index=filename_patient_open+filename_patient_close)
df_patient_delta.to_csv('patient_delta.csv',index=True)
df_patient_theta.to_csv('patient_theta.csv',index=True)
df_patient_alpha1.to_csv('patient_alpha1.csv',index=True)
df_patient_alpha2.to_csv('patient_alpha2.csv',index=True)
df_patient_beta.to_csv('patient_beta.csv',index=True)
df_patient_gamma.to_csv('patient_gamma.csv',index=True)
