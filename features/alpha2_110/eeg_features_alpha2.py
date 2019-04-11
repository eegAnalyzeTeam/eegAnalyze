import numpy as np
import mne
import check_file
import os
import pandas as pd
from multiprocessing import Process


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


def handle_badchannel(raw, badchannels):
    raw.load_data()
    raw.info['bads'] = badchannels
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    raw = raw.filter(10, 12)
    raw.resample(48, npad='auto')
    print('success resample')
    raw = raw.get_data(picks)
    raw = np.array(raw).T

    return raw


def control_thread_entity(raw, bad_channels, tsfresh_data, counter):
    temp_raw = handle_badchannel(raw, bad_channels)
    time = 0.0
    print(len(temp_raw))
    for x in temp_raw:
        x = list(x)
        x.insert(0, time)
        x.insert(0, counter)
        time += 0.01
        x.append('0')
        tsfresh_data.loc[len(tsfresh_data)] = x
    tsfresh_data.to_csv('control_data_' + str(counter) + '.csv')
    print(counter)


def patient_thread_entity(raw, bad_channels, tsfresh_data, counter):
    temp_raw = handle_badchannel(raw, bad_channels)
    time = 0.0
    print(len(temp_raw))
    for x in temp_raw:
        x = list(x)
        x.insert(0, time)
        x.insert(0, counter)
        time += 0.01
        x.append('1')
        tsfresh_data.loc[len(tsfresh_data)] = x
    tsfresh_data.to_csv('patient_data_' + str(counter) + '.csv')
    print(counter)


def save_csv_thread(control_raw, patient_raw, channel_names, bad_channels):
    columns = channel_names.copy()
    columns.insert(0, 'time')
    columns.insert(0, 'id')
    columns = columns[:-1]
    columns.append('y')
    tsfresh_data = pd.DataFrame(columns=columns)

    counter = 0

    threads = []
    for (eid, raw) in control_raw.items():
        t1 = Process(target=control_thread_entity, args=(raw, bad_channels, tsfresh_data, counter))
        threads.append(t1)
        counter += 1

    for (eid, raw) in patient_raw.items():
        t1 = Process(target=patient_thread_entity, args=(raw, bad_channels, tsfresh_data, counter))
        threads.append(t1)
        counter += 1

    i = 0
    for x in threads:
        i += 1
        x.start()
        if i % 10 == 0:
            x.join()

    x.join()


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

    tsfresh_data.to_csv('features_change/tsfresh_data_alpha2.csv')


def read_file(filePath):
    control_raw, patient_raw = read_data(filePath)
    print('read success')
    channel_names, bad_channels = raw_data_info(filePath)
    print('start save...')
    save_csv_thread(control_raw, patient_raw, channel_names, bad_channels)


def start():
    read_file('/home/rbai/eegData')
