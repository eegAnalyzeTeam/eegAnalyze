import mne
import os


def file_name(file_dir):
    files = []
    for root, dirs, fs in os.walk(file_dir):
        for file in fs:
            if str(file).find('EC') >= 0:
                files.append(root + str(file))
    return files


def get_patinet_control_name(name_list):
    control = []
    patient = []

    for x in name_list:
        if str(x).find('MDD') >= 0:
            patient.append(x)
        else:
            control.append(x)

    return control, patient


channel_names = ['EEG Fp1-LE', 'EEG F3-LE', 'EEG C3-LE', 'EEG P3-LE', 'EEG O1-LE', 'EEG F7-LE', 'EEG T3-LE',
                 'EEG T5-LE', 'EEG Fz-LE', 'EEG Fp2-LE', 'EEG F4-LE', 'EEG C4-LE', 'EEG P4-LE', 'EEG O2-LE',
                 'EEG F8-LE', 'EEG T4-LE', 'EEG T6-LE', 'EEG Cz-LE', 'EEG Pz-LE', 'EEG A2-A1']
bad_channel = ['EEG 23A-23R', 'EEG 24A-24R','STI 014']


# 3385168 = ['EEG Fp1-LE', 'EEG F3-LE', 'EEG C3-LE', 'EEG P3-LE', 'EEG O1-LE', 'EEG F7-LE', 'EEG T3-LE', 'EEG T5-LE',
#      'EEG Fz-LE', 'EEG Fp2-LE', 'EEG F4-LE', 'EEG C4-LE', 'EEG P4-LE', 'EEG O2-LE', 'EEG F8-LE', 'EEG T4-LE',
#      'EEG T6-LE', 'EEG Cz-LE', 'EEG Pz-LE', 'EEG A2-A1', 'EEG 23A-23R', 'EEG 24A-24R', 'STI 014']
# 4244171 = ['EEG Fp1-LE', 'EEG F3-LE', 'EEG C3-LE', 'EEG P3-LE', 'EEG O1-LE', 'EEG F7-LE', 'EEG T3-LE', 'EEG T5-LE',
#      'EEG Fz-LE', 'EEG Fp2-LE', 'EEG F4-LE', 'EEG C4-LE', 'EEG P4-LE', 'EEG O2-LE', 'EEG F8-LE', 'EEG T4-LE',
#      'EEG T6-LE', 'EEG Cz-LE', 'EEG Pz-LE', 'EEG A2-A1', 'EEG 23A-23R', 'EEG 24A-24R', 'STI 014']


def read_file(file_dir):
    files = file_name(file_dir)
    control_dir, patient_dir = get_patinet_control_name(files)


    count=0
    control_raw = []
    for dir in control_dir:
        raw = mne.io.read_raw_edf(dir, preload=True)
        raw = raw.pick_channels(channel_names)
        control_raw.append(raw)
        print(count)
        count+=1

    count=0
    patient_raw = []
    for dir in patient_dir:
        raw = mne.io.read_raw_edf(dir, preload=True)
        raw = raw.pick_channels(channel_names)
        patient_raw.append(raw)
        print(count)
        count+=1

    return control_raw, patient_raw


def start():
    control_raw,patient_raw=read_file('eegData_4244171/')

    print(len(control_raw))
    print(len(patient_raw))

    return control_raw,patient_raw




