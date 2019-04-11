import pandas as pd
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features

base_path = '/home/rbai/Multiclass/'


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
    extracted_features.to_csv(base_path + 'features/multiclass_60s_features_' + str(count) + '.csv')


# 将计算好的特征合并为一个csv
def get_sumcsv():
    base = pd.read_csv(base_path + 'features/multiclass_60s_features_' + str(0) + '.csv')

    for count in range(1, 59):
        if count == 12:
            continue
        base = base.append(pd.read_csv(base_path + 'features/multiclass_60s_features_' + str(count) + '.csv'))
        print(count)

    base.to_csv(base_path + 'multiclass_60s_features.csv', index=False)


# 入口函数，对每一个样本计算特征
def get_features():
    for count in range(59):
        get_feature_entity(base_path + 'data/multiclass_60s_data_' + str(count) + '.csv', count)


def start():
    get_features()
    get_sumcsv()
