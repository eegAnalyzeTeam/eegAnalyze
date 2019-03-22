import pandas as pd
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, select_features
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.externals import joblib
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

base_path = '../Multiclass/'


# 从文件里读取y
def get_y():
    y_csv_data = np.loadtxt(base_path + 'multiclass_60s_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)

    print(y)
    print(len(y))
    return y


# tsfresh select_features
def _select_features(extracted_features_name=base_path + 'multiclass_60s_features.csv'):
    y = get_y()

    # 全部特征
    extracted_features = pd.read_csv(base_path + extracted_features_name)

    # del extracted_features['Unnamed: 0']
    print(extracted_features)
    print('select start...')

    # 选取较相关的特征
    # 可选属性 fdr_level = 0.05 ?
    features_filtered = select_features(extracted_features, y, n_jobs=1, fdr_level=6, ml_task='classification')
    print(features_filtered)
    features_filtered.to_csv(base_path + 'tsfresh_filteredFeatures.csv')
    print('select end')


#  sklearn 线性选取特征
def test_sklearn_SelectFromModel(extracted_features_name=base_path + 'multiclass_60s_features.csv'):
    y = get_y()

    # 全部特征
    extracted_features = pd.read_csv(base_path + extracted_features_name)
    # 除掉 id
    del extracted_features['id']
    # del extracted_features['Unnamed: 0']

    cols = extracted_features.columns.values.tolist()
    print('select start...')

    y = np.array(np.array(y).tolist())
    extracted_features_arr = np.array(extracted_features)
    print(extracted_features)
    print(y)
    lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(extracted_features_arr, y)
    res = SelectFromModel(lsvc, prefit=True)
    features_filtered = res.transform(extracted_features_arr)
    cols = get_cols(cols, res.get_support())
    print(np.array(features_filtered))

    # # 获取列名？
    # res_col = []
    # arr = np.array(features_filtered).T
    # for i in arr:
    #     for indexs in extracted_features.columns:
    #         if list(extracted_features[indexs]) == list(i):
    #             res_col.append(indexs)
    #             break
    df = pd.DataFrame(features_filtered, columns=cols)
    df.to_csv(base_path + 'test_sklearn_SelectFromModel.csv')


#  sklearn 生成树选取特征
def test_sklearn_ExtraTreesClassifier(extracted_features_name=base_path + 'multiclass_60s_features.csv'):
    y = get_y()

    # 全部特征
    extracted_features = pd.read_csv(base_path + extracted_features_name)
    # 除掉 id
    del extracted_features['id']
    # del extracted_features['Unnamed: 0']

    cols = extracted_features.columns.values.tolist()
    print('select start...')

    y = np.array(np.array(y).tolist())
    extracted_features_arr = np.array(extracted_features)
    print(extracted_features)
    print(y)
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=4)
    clf = clf.fit(extracted_features_arr, y)
    res = SelectFromModel(clf, prefit=True)
    features_filtered = res.transform(extracted_features_arr)
    cols = get_cols(cols, res.get_support())
    print(np.array(features_filtered))

    df = pd.DataFrame(features_filtered, columns=cols)
    df.to_csv(base_path + 'test_sklearn_ExtraTreesClassifier.csv')


# 根据结果反馈，获取选择的特征名字
def get_cols(x, y):
    cols = []
    for i in range(len(y)):
        if y[i]:
            cols.append(x[i])
    return cols


#  sklearn 去除低方差
def test_sklearn_VarianceThreshold(extracted_features_name=base_path + 'multiclass_60s_features.csv'):
    y = get_y()

    # 全部特征
    extracted_features = pd.read_csv(base_path + extracted_features_name)
    # 除掉 id
    del extracted_features['id']
    # del extracted_features['Unnamed: 0']

    cols = extracted_features.columns.values.tolist()
    print('select start...')

    y = np.array(np.array(y).tolist())
    extracted_features_arr = np.array(extracted_features)
    print(extracted_features)
    print(y)
    res = VarianceThreshold(threshold=(.6 * (1 - .6)))
    features_filtered = res.fit_transform(extracted_features_arr)
    cols = get_cols(cols, res.fit(extracted_features_arr).get_support())
    print(np.array(features_filtered))

    df = pd.DataFrame(features_filtered, columns=cols)
    df.to_csv(base_path + 'test_sklearn_VarianceThreshold.csv')


# tsfresh 按照fdr选取特征
def test_select_features_VarianceThreshold(extracted_features_name=base_path + 'test_sklearn_VarianceThreshold.csv'):
    y = get_y()

    # 全部特征
    extracted_features = pd.read_csv(base_path + extracted_features_name)
    del extracted_features['Unnamed: 0']

    print(extracted_features)
    print('select start...')

    # 选取较相关的特征
    # 可选属性 fdr_level = 0.05 ?
    features_filtered = select_features(extracted_features, y, n_jobs=1, fdr_level=6, ml_task='classification')
    print(features_filtered)
    features_filtered.to_csv(base_path + 'select_features_VarianceThreshold.csv')
    print('select end')


# 入口主函数
def start():
    print('filter')
    _select_features()

    print('linear')
    test_sklearn_SelectFromModel()

    print('tree')
    test_sklearn_ExtraTreesClassifier()

    print('varianceThreshold')
    test_sklearn_VarianceThreshold()
    test_select_features_VarianceThreshold()


start()
