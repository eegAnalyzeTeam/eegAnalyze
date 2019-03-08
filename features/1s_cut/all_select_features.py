import pandas as pd
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features, select_features
from sklearn.feature_selection import SelectFromModel,VarianceThreshold,SelectKBest,chi2
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.externals import joblib
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# 从文件中读取y
def get_y():
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]

    print(y)
    print(len(y))
    return y


# 从文件读取feature,并使用tsfresh库的函数选取特征
def _select_features(extracted_features):
    y=get_y()

    # del extracted_features['Unnamed: 0']
    print(extracted_features)
    print('select start...')

    # 选取较相关的特征
    # 可选属性 fdr_level = 0.05 ?
    features_filtered = select_features(extracted_features, y ,n_jobs=1,fdr_level =0.0001,ml_task='classification')
    print(features_filtered)
    features_filtered.to_csv('tsfresh_filteredFeatures.csv')
    print('select end')


# sklearn库中 SelectFromModel， 线性特征选区
def test_sklearn_SelectFromModel(extracted_features):
    y=get_y()

    # 除掉 id
    del extracted_features['id']
    # del extracted_features['Unnamed: 0']

    cols=extracted_features.columns.values.tolist()
    print('select start...')

    y = np.array(np.array(y).tolist())
    extracted_features_arr = np.array(extracted_features)
    print(extracted_features)
    print(y)
    lsvc = LinearSVC(C=0.1, penalty="l1", dual=False).fit(extracted_features_arr, y)
    res = SelectFromModel(lsvc, prefit=True)
    features_filtered=res.transform(extracted_features_arr)
    cols=get_cols(cols,res.get_support())
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
    df.to_csv('test_sklearn_SelectFromModel.csv')


# sklearn 库中 ExtraTreesClassifier，生成树特征选取
def test_sklearn_ExtraTreesClassifier(extracted_features):
    y=get_y()

    # 除掉 id
    if 'id' in extracted_features.columns.values.tolist():
        del extracted_features['id']
    # del extracted_features['Unnamed: 0']

    cols=extracted_features.columns.values.tolist()
    print('select start...')

    y = np.array(np.array(y).tolist())
    extracted_features_arr = np.array(extracted_features)
    print(extracted_features)
    print(y)
    clf = ExtraTreesClassifier(n_estimators=4, max_depth=4)
    clf = clf.fit(extracted_features_arr, y)
    res=SelectFromModel(clf, prefit=True)
    features_filtered = res.transform(extracted_features_arr)
    cols=get_cols(cols,res.get_support())
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
    df.to_csv('test_sklearn_ExtraTreesClassifier_4.csv')


# 将留下的特征抽取出来
def get_cols(x,y):
    cols=[]
    for i in range(len(y)):
        if y[i]:
            cols.append(x[i])
    return cols


# sklearn库中的 VarianceThreshold，删除低方差的特征
def test_sklearn_VarianceThreshold(extracted_features):
    y=get_y()

    # 除掉 id
    if 'id' in extracted_features.columns.values.tolist():
        del extracted_features['id']
    # del extracted_features['Unnamed: 0']

    cols=extracted_features.columns.values.tolist()
    print('select start...')

    y = np.array(np.array(y).tolist())
    extracted_features_arr = np.array(extracted_features)
    print(extracted_features)
    print(y)
    res = VarianceThreshold(threshold=(.6 * (1 - .6)))
    features_filtered=res.fit_transform(extracted_features_arr)
    cols=get_cols(cols,res.fit(extracted_features_arr).get_support())
    print(np.array(features_filtered))

    df = pd.DataFrame(features_filtered, columns=cols)
    df.to_csv('test_sklearn_VarianceThreshold.csv')


# 利用tsfresh库select_features函数选取特征（先删除低方差后）
# 在test_sklearn_VarianceThreshold函数后调用
def test_select_features_VarianceThreshold(extracted_features_name='test_sklearn_VarianceThreshold.csv'):
    y=get_y()

    # 全部特征
    extracted_features = pd.read_csv(extracted_features_name)
    del extracted_features['Unnamed: 0']

    print(extracted_features)
    print('select start...')

    # 选取较相关的特征
    # 可选属性 fdr_level = 0.05 ?
    features_filtered = select_features(extracted_features, y, n_jobs=1, fdr_level=0.01,ml_task='classification')
    print(features_filtered)
    features_filtered.to_csv('select_features_VarianceThreshold.csv')
    print('select end')


# 入口函数，分别以几种不同的方式选取计算好的特征
def start(extracted_features_name='tsfresh_extractedFeatures.csv'):

    print('start ...')
    extracted_features = pd.read_csv(extracted_features_name)

    print('filter')
    _select_features(extracted_features)

    print('linear')
    test_sklearn_SelectFromModel(extracted_features)

    print('tree')
    test_sklearn_ExtraTreesClassifier(extracted_features)

    print('varianceThreshold')
    test_sklearn_VarianceThreshold(extracted_features)
    test_select_features_VarianceThreshold()


start()
