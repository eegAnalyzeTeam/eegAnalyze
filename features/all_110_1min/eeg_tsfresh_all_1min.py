import numpy as np
import mne
import check_file
import os, sys
import pandas as pd
from multiprocessing import Process
import thread_cal_features
# import eeg_tsfresh_calcFeatures
import threading


def raw_data_info(filePath):
    raw = mne.io.read_raw_brainvision(filePath + '/health_control/eyeclose/jkdz_cc_20180430_close.vhdr',
                                      preload=True)
    # channel_names = raw.info['ch_names']
    print()
    channel_names = []
    for i in raw.info['ch_names']:
        if i != 'Oz':
            if i != 'ECG':
                channel_names.append(i)

    bad_channels = ['Oz', 'ECG']
    return channel_names, bad_channels


def troublesome_data(filePath):
    control_q = []
    patient_q = []
    for dirpath, dirs, files in os.walk(filePath):

        if 'eyeclose' in dirpath and 'health_control' in dirpath:
            # health control group
            for fname in files:
                if '.vhdr' in fname:
                    id_control = fname[:-5]
                    vmrkf, eegf = check_file.get_vhdr_info(dirpath + '/' + fname)
                    if vmrkf == eegf and vmrkf == id_control:
                        print('OK')
                    else:
                        control_q.append(id_control)

        elif 'eyeclose' in dirpath and 'mdd_patient' in dirpath:
            # mdd group
            for fname in files:
                if '.vhdr' in fname:
                    id_patient = fname[:-5]
                    vmrkf, eegf = check_file.get_vhdr_info(dirpath + '/' + fname)
                    if vmrkf == eegf and vmrkf == id_patient:
                        print('OK')
                    else:
                        patient_q.append(id_patient)

    return control_q, patient_q


def read_data(filePath):
    # q contains troublesome eeg files. skip them for now
    control_q, patient_q = troublesome_data(filePath)
    # q = ['njh_after_pjk_20180725_close.vhdr', 'ccs_yb_20180813_close.vhdr', 'njh_before_pjk_20180613_close.vhdr', 'ccs_before_wjy_20180817_close.vhdr', 'ccs_after_csx_20180511_close.vhdr']
    print(patient_q)
    print('---------===========-----------')
    control_raw = {}
    patient_raw = {}

    for dirpath, dirs, files in os.walk(filePath):

        if 'eyeclose' in dirpath and 'health_control' in dirpath:
            # health control group
            for fname in files:
                if '.vhdr' in fname and fname not in control_q:
                    id_control = fname[:-5]

                    raw = mne.io.read_raw_brainvision(dirpath + '/' + fname, preload=False)
                    if len(raw.info['ch_names']) == 65:
                        raw.set_montage(mne.channels.read_montage("standard_1020"))
                        control_raw[id_control] = raw
                    else:
                        print("Abnormal data with " + str(len(raw.info['ch_names'])) + " channels. id=" + id_control)

        elif 'eyeclose' in dirpath and 'mdd_patient' in dirpath:
            # mdd group
            for fname in files:
                if '.vhdr' in fname and fname[:-5] not in patient_q:
                    id_patient = fname[:-5]

                    raw = mne.io.read_raw_brainvision(dirpath + '/' + fname, preload=False)

                    if len(raw.info['ch_names']) == 65:
                        raw.set_montage(mne.channels.read_montage("standard_1020"))
                        patient_raw[id_patient] = raw
                    else:
                        print("Abnormal data with " + str(len(raw.info['ch_names'])) + " channels. id=" + id_patient)

    return control_raw, patient_raw


def handle_badchannel(raw, badchannels, counter):
    raw.load_data()
    raw = raw.filter(None, 40)
    print(len(raw))
    print('filter success')
    raw.resample(160, npad='auto')
    print(len(raw))
    if len(raw)>100000 or len(raw)<19000:
        print(counter)
        fileObject = open(str(counter)+'.txt', 'w')
        fileObject.write(str(counter))
        fileObject.close()
    print('success resample')
    raw.info['bads'] = badchannels
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')


    raw = raw.get_data(picks,start=9500,stop=19000)
    raw=np.array(raw).T

    return raw


def save_csv(control_raw, patient_raw, channel_names, bad_channels):
    columns = channel_names.copy()
    columns.insert(0, 'time')
    columns.insert(0, 'id')
    columns = columns[:-1]
    columns.append('y')
    tsfresh_data = pd.DataFrame(columns=columns)

    counter = 0
    for (eid, raw) in control_raw.items():
        temp_raw = handle_badchannel(raw, bad_channels)
        time = 0.0
        for x in temp_raw:
            x = list(x)
            x.insert(0, time)
            x.insert(0, counter)
            time += 0.01
            x.append('0')
            tsfresh_data.loc[len(tsfresh_data)] = x
        print(counter)
        counter += 1

    for (eid, raw) in patient_raw.items():
        temp_raw = handle_badchannel(raw, bad_channels)
        time = 0.0
        for x in temp_raw:
            x = list(x)
            x.insert(0, time)
            x.insert(0, counter)
            time += 0.01
            x.append('1')
            tsfresh_data.loc[len(tsfresh_data)] = x
        counter += 1
        print(counter)

    tsfresh_data.to_csv('features_48/tsfresh_data_48.csv')


def control_thread_entity(raw, bad_channels, tsfresh_data, counter):
    temp_raw_arr = handle_badchannel(raw, bad_channels, counter)
    for temp_raw in temp_raw_arr:
        time = 0.0
        tsfresh_data_copy=tsfresh_data.copy()
        print(len(temp_raw))
        for x in temp_raw:
            x = list(x)
            x.insert(0, time)
            x.insert(0, counter)
            time += 0.01
            x.append('1')
            tsfresh_data_copy.loc[len(tsfresh_data_copy)] = x
        tsfresh_data_copy.to_csv('control_data_' + str(counter) + '.csv')
        counter += 1
        print(counter)
    return counter


def patient_thread_entity(raw, bad_channels, tsfresh_data, counter):
    temp_raw_arr = handle_badchannel(raw, bad_channels, counter)
    for temp_raw in temp_raw_arr:
        time = 0.0
        tsfresh_data_copy=tsfresh_data.copy()
        print(len(temp_raw))
        for x in temp_raw:
            x = list(x)
            x.insert(0, time)
            x.insert(0, counter)
            time += 0.01
            x.append('0')
            tsfresh_data_copy.loc[len(tsfresh_data_copy)] = x
        tsfresh_data_copy.to_csv('patient_data_' + str(counter) + '.csv')
        counter += 1
        print(counter)
    return counter


def save_csv_thread(control_raw, patient_raw, channel_names, bad_channels):
    columns = channel_names.copy()
    columns.insert(0, 'time')
    columns.insert(0, 'id')
    columns = columns[:-1]
    columns.append('y')
    tsfresh_data = pd.DataFrame(columns=columns)

    counter = 0
    person_count=0

    for (eid, raw) in control_raw.items():
        # if person_count == 0 or person_count ==15:
        #     person_count += 1
        #     continue
        if len(raw)>2000000:
            fileObject = open(str(person_count)+'.txt', 'w')
            fileObject.write(str(person_count))
            fileObject.close()
            continue
        counter = control_thread_entity(raw, bad_channels, tsfresh_data, counter)
        person_count+=1
        print(person_count)

    for (eid, raw) in patient_raw.items():
        # if person_count == 77:
        #     person_count += 1
        #     continue
        if len(raw)<400000:
            fileObject = open(str(person_count)+'.txt', 'w')
            fileObject.write(str(person_count))
            fileObject.close()
            continue
        counter = patient_thread_entity(raw, bad_channels, tsfresh_data, counter)
        person_count+=1
        print(person_count)




def read_file(filePath):
    control_raw, patient_raw = read_data(filePath)
    print('read success')
    channel_names, bad_channels = raw_data_info(filePath)
    print('start save...')
    save_csv_thread(control_raw, patient_raw, channel_names, bad_channels)


read_file('/home/rbai/eegData')

# read_file('/home/public2/eegData')

# eeg_tsfresh_calcFeatures.get_features('tsfresh_data.csv')
thread_cal_features.get_features_thread()