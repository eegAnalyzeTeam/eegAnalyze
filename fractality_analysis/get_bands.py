# 将eeg信号划分为5个频带

import mne
import read_file

right_brain = ['FP2', 'F4', 'F8']
left_brain = ['FP1', 'F3', 'F7']
brain = ['FP1', 'F3', 'F7', 'FP2', 'F4', 'F8', 'Fz']


def get_filter(raw):
    res = {}
    res['full']=raw

    raw_temp = raw.copy()
    res['delta'] = raw_temp.filter(1, 4, fir_design='firwin')

    raw_temp = raw.copy()
    res['theta'] = raw_temp.filter(4, 8, fir_design='firwin')

    raw_temp = raw.copy()
    res['alpha'] = raw_temp.filter(8, 12, fir_design='firwin')

    raw_temp = raw.copy()
    res['beta'] = raw_temp.filter(15, 30, fir_design='firwin')

    raw_temp = raw.copy()
    res['gamma'] = raw_temp.filter(30, 70, fir_design='firwin')

    return res


def handle_raw(raw):
    # # 去掉眼电
    # raw=raw.filter(1, None, fir_design="firwin")
    # ica = mne.preprocessing.ICA(method='extended-infomax', random_state = 1)
    # raw = ica.fit(raw)
    raw = raw.load_data()
    if len(raw.get_data()[0]) > 6000000:
        return None
    raw = raw.pick_channels(brain)
    raw = raw.resample(256, npad="auto")
    res = get_filter(raw)
    return res


def get_bands():
    control_raw, patient_raw, channel_names = read_file.read_file()
    control_res = []
    count = 0
    for (eid, raw) in control_raw.items():
        temp = handle_raw(raw)
        if temp is None:
            print('too large')
            continue
        control_res.append(temp)
        print('control: ' + str(count))
        count += 1

    patient_res = []
    count = 0
    for (eid, raw) in patient_raw.items():
        temp = handle_raw(raw)
        if temp is None:
            print('too large')
            continue
        control_res.appe
        patient_res.append(temp)
        print('patient: ' + str(count))
        count += 1

    return control_res, patient_res
