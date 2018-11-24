import mne
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd

def raw_data_info():
	raw = mne.io.read_raw_brainvision('/home/public2/eegData/health_control/eyeclose/jkdz_cc_20180430_close.vhdr',preload=True)
	channel_names = raw.info['ch_names']
	bad_channels = []
	return channel_names, bad_channels
def calculate_eeg_psd(raw):
    #data_path='D:\health_control\eyeopen'
    #fname=data_path+'\jkdz_dhl_20180411_open.vhdr'
    # print(__doc__)
    # raw=mne.io.read_raw_brainvision(fname,preload=True)
    psd_subfreq={}

    psd_subfreq['delta'],freqs1=mne.time_frequency.psd_welch(raw,fmin=0.5,fmax=4,n_fft=2048,n_jobs=1)
    psd_subfreq['theta'],freqs2=mne.time_frequency.psd_welch(raw,fmin=4,fmax=7,n_fft=2048,n_jobs=1)
    psd_subfreq['alpha1'],freqs3=mne.time_frequency.psd_welch(raw,fmin=8,fmax=10,n_fft=2048,n_jobs=1)
    psd_subfreq['alpha2'] ,freqs4=mne.time_frequency.psd_welch(raw,fmin=10,fmax=12,n_fft=2300,n_jobs=1)
    psd_subfreq['beta'],freqs5=mne.time_frequency.psd_welch(raw,fmin=13,fmax=30,n_fft=256,n_jobs=1)
    psd_subfreq['gamma'],freqs6=mne.time_frequency.psd_welch(raw,fmin=30,fmax=40,n_fft=512,n_jobs=1)
    
    return psd_subfreq

def eeg_psd(control_raw, patient_raw):
	channel_names, bad_channels = raw_data_info()
	columns = channel_names.copy()
	columns.insert(0, 'id')
	columns = columns[:-1]
	df_c_delta = pd.DataFrame(columns = columns)
	df_c_theta = pd.DataFrame(columns = columns)
	df_c_alpha1 = pd.DataFrame(columns = columns)
	df_c_alpha2 = pd.DataFrame(columns = columns)
	df_c_beta = pd.DataFrame(columns = columns)
	df_c_gamma = pd.DataFrame(columns = columns)

	df_p_delta = pd.DataFrame(columns = columns)
	df_p_theta = pd.DataFrame(columns = columns)
	df_p_alpha1 = pd.DataFrame(columns = columns)
	df_p_alpha2 = pd.DataFrame(columns = columns)
	df_p_beta = pd.DataFrame(columns = columns)
	df_p_gamma = pd.DataFrame(columns = columns)

	counter = 0
	for (eid, raw) in control_raw.items():
		psd_subfreq = calculate_eeg_psd(raw)
		print('control: ' + str(counter))
		counter += 1
		temp = list(psd_subfreq['delta'].copy())
		temp = [t[0] for t in psd_subfreq['delta']]
		temp.insert(0, eid)

		df_c_delta.loc[len(df_c_delta)] = temp

		temp = list(psd_subfreq['theta'].copy())
		temp = [t[0] for t in psd_subfreq['theta']]
		temp.insert(0, eid)
		df_c_theta.loc[len(df_c_theta)] = temp

		temp = list(psd_subfreq['alpha1'].copy())
		temp = [t[0] for t in psd_subfreq['alpha1']]
		temp.insert(0, eid)
		df_c_alpha1.loc[len(df_c_alpha1)] = temp

		temp = list(psd_subfreq['alpha2'].copy())
		temp = [t[0] for t in psd_subfreq['alpha2']]
		temp.insert(0, eid)
		df_c_alpha2.loc[len(df_c_alpha2)] = temp

		temp = list(psd_subfreq['beta'].copy())
		temp = [t[0] for t in psd_subfreq['beta']]
		temp.insert(0, eid)
		df_c_beta.loc[len(df_c_beta)] = temp

		temp = list(psd_subfreq['gamma'].copy())
		temp = [t[0] for t in psd_subfreq['gamma']]
		temp.insert(0, eid)
		df_c_gamma.loc[len(df_c_gamma)] = temp


	df_c_delta.to_csv('psd_c_delta.csv',index=True)
	df_c_theta.to_csv('psd_c_theta.csv',index=True)
	df_c_alpha1.to_csv('psd_c_alpha1.csv',index=True)
	df_c_alpha2.to_csv('psd_c_alpha2.csv',index=True)
	df_c_beta.to_csv('psd_c_beta.csv',index=True)
	df_c_gamma.to_csv('psd_c_gamma.csv',index=True)

	counter = 0
	for (eid, raw) in patient_raw.items():
		psd_subfreq = calculate_eeg_psd(raw)
		print('patient: ' + str(counter))
		counter += 1
		temp = list(psd_subfreq['delta'].copy())
		temp = [t[0] for t in psd_subfreq['delta']]
		temp.insert(0, eid)
		df_p_delta.loc[len(df_p_delta)] = temp

		temp = list(psd_subfreq['theta'].copy())
		temp = [t[0] for t in psd_subfreq['theta']]
		temp.insert(0, eid)
		df_p_theta.loc[len(df_p_theta)] = temp

		temp = list(psd_subfreq['alpha1'].copy())
		temp = [t[0] for t in psd_subfreq['alpha1']]
		temp.insert(0, eid)
		df_p_alpha1.loc[len(df_p_alpha1)] = temp

		temp = list(psd_subfreq['alpha2'].copy())
		temp = [t[0] for t in psd_subfreq['alpha2']]
		temp.insert(0, eid)
		df_p_alpha2.loc[len(df_p_alpha2)] = temp

		temp = list(psd_subfreq['beta'].copy())
		temp = [t[0] for t in psd_subfreq['beta']]
		temp.insert(0, eid)
		df_p_beta.loc[len(df_p_beta)] = temp

		temp = list(psd_subfreq['gamma'].copy())
		temp = [t[0] for t in psd_subfreq['gamma']]
		temp.insert(0, eid)
		df_p_gamma.loc[len(df_p_gamma)] = temp


	df_p_delta.to_csv('psd_p_delta.csv',index=True)
	df_p_theta.to_csv('psd_p_theta.csv',index=True)
	df_p_alpha1.to_csv('psd_p_alpha1.csv',index=True)
	df_p_alpha2.to_csv('psd_p_alpha2.csv',index=True)
	df_p_beta.to_csv('psd_p_beta.csv',index=True)
	df_p_gamma.to_csv('psd_p_gamma.csv',index=True)







