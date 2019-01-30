import pandas as pd
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, select_features
from multiprocessing import Process
import threading


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
    threads=[]

    for i in range(16, 30):
        temp='control_data_' + str(i) + '.csv'
        get_features(temp,i)
        # t1 = threading.Thread(target=get_features, args=(temp,i))
        # threads.append(t1)

    for i in range(30, 111):
        temp='patient_data_' + str(i) + '.csv'
        get_features(temp, i)
        # t1 = threading.Thread(target=get_features, args=(temp,i))
        # threads.append(t1)

    # i = 0
    # for x in threads:
    #     i += 1
    #     x.setDaemon(True)
    #     x.start()
    #     if i % 5 == 0:
    #         x.join()
    #         threads[i - 2].join()
    #         threads[i - 3].join()
    #         threads[i - 4].join()
    #         threads[i - 5].join()



get_features_thread()