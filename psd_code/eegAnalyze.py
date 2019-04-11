# This program is where the main function at.
# The program calls preprocessing functions, calculate psds for each
# person and use anova test to find the significant channels and sub-frequencies
import os

import mne

from psd_code import check_file
from psd_code import eeg_psd_anova
from psd_code import eeg_psd_csv
from psd_code import eeg_psd_plot


def troublesome_data(filePath):
    control_q = []
    patient_q = []
    for dirpath, _, files in os.walk(filePath):
        if 'eyeclose' in dirpath and 'health_control' in dirpath:
            # health control group
            for fname in files:
                if '.vhdr' in fname:
                    id_control = fname[:-5]
                    vmrkf, eegf = check_file.get_vhdr_info(dirpath + '/' + fname)
                    if vmrkf == eegf and vmrkf == id_control:
                        print('OK')
                    else:
                        # print('control: vhdr:' + id_control + ' vmrk: ' + vmrkf + ' eeg:' + eegf)
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
                        # print('patient: vhdr:' + id_patient + ' vmrk: ' + vmrkf + ' eeg:' + eegf)
                        patient_q.append(id_patient)

    return control_q, patient_q


def readData(filePath):
    # q contains troublesome eeg files. skip them for now
    control_q, patient_q = troublesome_data(filePath)
    # q = ['njh_after_pjk_20180725_close.vhdr', 'ccs_yb_20180813_close.vhdr', 'njh_before_pjk_20180613_close.vhdr', 'ccs_before_wjy_20180817_close.vhdr', 'ccs_after_csx_20180511_close.vhdr']
    print(patient_q)
    print('---------===========-----------')
    control_raw = {}
    patient_raw = {}

    for dirpath, _, files in os.walk(filePath):

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
                        print("Abnormal data with " +
                              str(len(raw.info['ch_names'])) + " channels. id=" + id_control)

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
                        print("Abnormal data with " +
                              str(len(raw.info['ch_names'])) + " channels. id=" + id_patient)

    return control_raw, patient_raw
    # return control_q, patient_q


def start():
    control_raw, patient_raw = readData('/home/rbai/eegData')

    eeg_psd_csv.eeg_psd(control_raw, patient_raw)

    eeg_psd_anova.psd_anova()

    eeg_psd_plot.plot_psd()

    print('control: ' + str(len(control_raw)))
    print('patient: ' + str(len(patient_raw)))
