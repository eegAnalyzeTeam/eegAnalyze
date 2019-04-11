import eeg_getData
import numpy as np
import mne
import csv

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

import pandas as pd

def handle_channel(raw):
    picks = mne.pick_types(raw.info, eeg=True)

    temp = len(raw)
    temp_res = []
    i = 0
    while temp - (512 * 3) >= 0:
        x = raw.get_data(picks, start=i, stop=i + (512 * 3))
        i += (512 * 3)
        temp -= (512 * 3)
        x = np.array(x).T
        temp_res.append(x)
        print(temp)

    return temp_res


def control_thread_entity(raw, columns, counter):
    temp_raw_arr = handle_channel(raw)
    for temp_raw in temp_raw_arr:
        time = 0.0
        print(len(temp_raw))
        fileread = open('data/control_data_' + str(counter) + '.csv', 'w', newline='')
        writer = csv.writer(fileread)
        writer.writerow(columns)
        for x in temp_raw:
            x = list(x)
            x.insert(0, time)
            x.insert(0, counter)
            time += 0.01
            x.append('0')
            writer.writerow(x)
        fileread.close
        counter += 1
        print(counter)
    return counter


def patient_thread_entity(raw, columns, counter):
    temp_raw_arr = handle_channel(raw)
    for temp_raw in temp_raw_arr:
        time = 0.0
        print(len(temp_raw))
        fileread = open('data/patient_data_' + str(counter) + '.csv', 'w', newline='')
        writer = csv.writer(fileread)
        writer.writerow(columns)
        for x in temp_raw:
            x = list(x)
            x.insert(0, time)
            x.insert(0, counter)
            time += 0.01
            x.append('1')
            writer.writerow(x)
        fileread.close
        counter += 1
        print(counter)
    return counter


def get_DataFrame(control_raw, patient_raw):
    columns = eeg_getData.channel_names
    columns = list(map(lambda x: x[4:], columns))

    columns.insert(0, 'time')
    columns.insert(0, 'id')
    columns.append('y')
    print(columns)

    counter = 0
    person_count = 0

    for raw in control_raw:
        counter = control_thread_entity(raw, columns, counter)
        person_count += 1
        print(person_count)

    for raw in patient_raw:
        counter = patient_thread_entity(raw, columns, counter)
        person_count += 1
        print(person_count)



def handle_y(y):
    y=y.drop_duplicates(subset=['id', 'y'], keep='first')
    y=y.reset_index(drop=True)
    y=y.iloc[:,-1]

    return y


# 有效特征
def get_features(file_name,count):
    csv_data = pd.read_csv(file_name)
    timeseries = csv_data.iloc[:, :-1]


    print('start getfeatures...')
    # 全部特征
    extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
    impute(extracted_features)
    print('start save ...')
    extracted_features.to_csv('data/tsfresh_extractedFeatures'+str(count)+'.csv')

    print(str(count)+'  end')


def get_features_thread():
    # threads=[]

    for i in range(0, 1420):
        try:
            temp='data/control_data_' + str(i) + '.csv'
            get_features(temp,i)
        except Exception:
            print(i)

    for i in range(1420, 2881):
        try:
            temp='data/patient_data_' + str(i) + '.csv'
            get_features(temp, i)
        except Exception:
            print(i)

# control_raw, patient_raw = eeg_getData.start()
# get_DataFrame(control_raw, patient_raw)
get_features_thread()