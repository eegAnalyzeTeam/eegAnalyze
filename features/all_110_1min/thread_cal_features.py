import pandas as pd
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, select_features
from multiprocessing import Process
import threading
import eeg_tsfresh_calcFeatures


def handle_y(y):
    y=y.drop_duplicates(subset=['id', 'y'], keep='first')
    y=y.reset_index(drop=True)
    y=y.iloc[:,-1]

    return y


# 有效特征
def get_features(file_name,count):
    csv_data = pd.read_csv(file_name)
    timeseries = csv_data.iloc[:, :-1]
    del timeseries['Unnamed: 0']
    y = csv_data[['id', 'y']]
    y = handle_y(y)

    print(timeseries)
    print(y)

    print('start getfeatures...')
    # 全部特征
    extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
    impute(extracted_features)
    extracted_features.to_csv('tsfresh_extractedFeatures'+str(count)+'.csv')
    print(str(count)+'  end')


def get_features_thread():

    for i in range(0, 30):
        try:
            temp='control_data_' + str(i) + '.csv'
            get_features(temp,i)
        except Exception:
            print(i)

    for i in range(30, 111):
        try:
            temp='patient_data_' + str(i) + '.csv'
            get_features(temp, i)
        except Exception:
            print(i)



def read_allcut_extracedFeatures():
    lack_alpha2=[0,15,77]
    base = pd.read_csv('tsfresh_extractedFeatures' + str(0) + '.csv')
    for i in range(1, 111):
        if i in lack_alpha2:
            continue
        temp = pd.read_csv('tsfresh_extractedFeatures' + str(i) + '.csv')
        base = base.append(temp)
        print('alpha2 '+str(i))

    base.to_csv('extracted_features.csv')



get_features_thread()

read_allcut_extracedFeatures()
eeg_tsfresh_calcFeatures.start()