import mne
import numpy as np
from matplotlib import pyplot as plt

def eeg_psd(fname):
    #data_path='D:\health_control\eyeopen'
    #fname=data_path+'\jkdz_dhl_20180411_open.vhdr'
    print(__doc__)
    raw=mne.io.read_raw_brainvision(fname,preload=True)
    raw.set_montage(mne.channels.read_montage("standard_1020"))
    
    duration=10
    event_id=1
    events = mne.make_fixed_length_events(raw, event_id, duration=duration)
    print(events)
    epochs = mne.Epochs(raw,events=events,tmin=0, tmax=10,baseline=(None, 0),verbose=True,
                        preload=True)
    #epochs.plot(scalings='auto', block=True)
    print(epochs)
    psd_all,freqs=mne.time_frequency.psd_multitaper(epochs,fmin=0,fmax=50,proj=True,n_jobs=1)


    raw_delta=raw.copy()
    raw_theta=raw.copy()
    raw_alpha1=raw.copy()
    raw_alpha2=raw.copy()
    raw_beta=raw.copy()
    raw_gamma=raw.copy()
    raw_delta.filter(0.5,4,fir_design='firwin')
    raw_theta.filter(4,7,fir_design='firwin')
    raw_alpha1.filter(8,10,fir_design='firwin')
    raw_alpha2.filter(10,12,fir_design='firwin')
    raw_beta.filter(13,30,fir_design='firwin')
    raw_gamma.filter(30,40,fir_design='firwin')
    
    epochs_delta = mne.Epochs(raw_delta,events=events,tmin=0, tmax=10,baseline=(None, 0),verbose=True,
                              preload=True)
    print(epochs_delta)
    del raw_delta
    psds_delta,freqs=mne.time_frequency.psd_multitaper(epochs_delta,fmin=0.5,fmax=4,proj=True,n_jobs=1)
    del epochs_delta
    
    epochs_theta = mne.Epochs(raw_theta,events=events,tmin=0, tmax=10,baseline=(None, 0),verbose=True,
                              preload=True)
    print(epochs_theta)
    del raw_theta
    psds_theta,freqs=mne.time_frequency.psd_multitaper(epochs_theta,fmin=4,fmax=7,proj=True,n_jobs=1)
    del epochs_theta
    
    
    epochs_alpha1 = mne.Epochs(raw_alpha1,events=events,tmin=0, tmax=10,baseline=(None, 0),verbose=True,
                               preload=True)
    print(epochs_alpha1)
    del raw_alpha1
    psds_alpha1,freqs=mne.time_frequency.psd_multitaper(epochs_alpha1,fmin=8,fmax=10,proj=True,n_jobs=1)
    del epochs_alpha1
        
    
    epochs_alpha2 = mne.Epochs(raw_alpha2 ,events=events,tmin=0, tmax=10,baseline=(None, 0),verbose=True,
                               preload=True)
    print(epochs_alpha2 )
    del raw_alpha2
    psds_alpha2 ,freqs=mne.time_frequency.psd_multitaper(epochs_alpha2 ,fmin=10,fmax=12,proj=True,n_jobs=1)
    del epochs_alpha2
    
    
    epochs_beta = mne.Epochs(raw_beta,events=events,tmin=0, tmax=10,baseline=(None, 0),verbose=True,
                             preload=True)
    print(epochs_beta)
    del raw_beta
    psds_beta,freqs=mne.time_frequency.psd_multitaper(epochs_beta,fmin=13,fmax=30,proj=True,n_jobs=1)
    del epochs_beta
    
    epochs_gamma = mne.Epochs(raw_gamma,events=events,tmin=0, tmax=10,baseline=(None, 0),verbose=True,
                              preload=True)
    print(epochs_gamma)
    del raw_gamma
    psds_gamma,freqs=mne.time_frequency.psd_multitaper(epochs_gamma,fmin=30,fmax=40,proj=True,n_jobs=1)
    del epochs_gamma
    
    psd_subfreq={}
    psd_subfreq['delta']=psds_delta
    psd_subfreq['theta']=psds_theta
    psd_subfreq['alpha1']=psds_alpha1
    psd_subfreq['alpha2']=psds_alpha2
    psd_subfreq['beta']=psds_beta
    psd_subfreq['gamma']=psds_gamma
    
    return psd_all,psd_subfreq
