import pandas as pd
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, select_features
import os

base_path = '../Multiclass/'


# 根据传入的参数tsfresh计算特征
def get_feature_entity(file_name, count):
    csv_data = pd.read_csv(base_path + file_name)
    timeseries = csv_data.iloc[:, :-1]

    print(timeseries)

    print(str(count) + 'count start getfeatures...')
    # 全部特征
    extracted_features = extract_features(timeseries, column_id="id", column_sort="time")
    impute(extracted_features)
    print(str(count) + 'start save ...')
    extracted_features.to_csv(base_path + 'features/multiclass_180s_features_' + str(count) + '.csv')


# 将计算好的特征合并为一个csv
def get_sumcsv():
    path = '/home/rbai/Multiclass/features'  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    base = pd.read_csv(base_path + files[0])
    count = 0

    for file in files:
        if count == 0:
            count += 1
            continue
        temp = pd.read_csv(base_path + file)
        base = base.append(temp)
        count += 1
        print(count)

    base.to_csv(base_path + 'multiclass_180s_features.csv', index=False)


# 入口函数，对每一个样本计算特征
def get_features():
    path = '/home/rbai/Multiclass/data'  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称

    count = 0
    for file in files:
        get_feature_entity('data/' + file, count)
        count += 1

    get_sumcsv()


get_features()
