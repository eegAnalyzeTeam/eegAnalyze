import pandas as pd
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features


def handle_y(y):
    y = y.drop_duplicates(subset=['id', 'y'], keep='first')
    y = y.reset_index(drop=True)
    y = y.iloc[:, -1]

    return y


# 有效特征
def get_features(file_name, count):
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
    extracted_features.to_csv('tsfresh_extractedFeatures' + str(count) + '.csv')
    print(str(count) + '  end')


def get_features_thread():
    _error = []

    for i in range(0, 30):
        try:
            temp = 'control_data_' + str(i) + '.csv'
            get_features(temp, i)
        except Exception:
            print(str(i) + ' is error')
            _error.append(str(i))

    for i in range(30, 111):
        try:
            temp = 'patient_data_' + str(i) + '.csv'
            get_features(temp, i)
        except Exception:
            print(str(i) + ' is error')
            _error.append(str(i))

    print(_error)


def start():
    get_features_thread()
