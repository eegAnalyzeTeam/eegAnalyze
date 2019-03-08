import pandas as pd
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features


# 针对每一个csv文件，用tsfresh库计算全部特征
def get_features(file_name,count):
    csv_data = pd.read_csv(file_name)
    timeseries = csv_data.iloc[:, :-1]


    print('start getfeatures...')
    # 全部特征
    extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
    impute(extracted_features)
    print('start save ...')
    extracted_features.to_csv('tsfresh_extractedFeatures'+str(count)+'.csv')

    print(str(count)+'  end')


# 一个入口函数，两个循环分别计算正常人和病人的全部特征
def get_features_thread():
    # threads=[]

    # 正常人
    for i in range(0, 5774):
        try:
            temp='control_data_' + str(i) + '.csv'
            get_features(temp,i)
        except Exception:
            print(i)

    # 病人
    for i in range(23003, 24363):
        try:
            temp='patient_data_' + str(i) + '.csv'
            get_features(temp, i)
        except Exception:
            print(i)

    # for i in range(1754, 1913):
    #     try:
    #         temp = 'control_data_' + str(i) + '.csv'
    #         get_features(temp, i)
    #     except Exception:
    #         print(i)
    #
    # for i in range(7021, 8082):
    #     try:
    #         temp = 'patient_data_' + str(i) + '.csv'
    #         get_features(temp, i)
    #     except Exception:
    #         print(i)



get_features_thread()
