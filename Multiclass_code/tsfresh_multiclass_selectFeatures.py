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


base_path = '../Multiclass/'

def get_y():
    y_csv_data = np.loadtxt(base_path + 'multiclass_180s_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)

    print(y)
    print(len(y))
    return y


# 从文件读取feature,在已经保存全部特征的情况下使用
def _select_features(extracted_features_name='multiclass_180s_data.csv'):
    y=get_y()

    # 全部特征
    extracted_features = pd.read_csv(base_path+extracted_features_name)

    # del extracted_features['Unnamed: 0']
    print(extracted_features)
    print('select start...')

    # 选取较相关的特征
    # 可选属性 fdr_level = 0.05 ?
    features_filtered = select_features(extracted_features, y ,n_jobs=1,fdr_level =0.001,ml_task='classification')
    print(features_filtered)
    features_filtered.to_csv(base_path+'tsfresh_filteredFeatures.csv')
    print('select end')


# test sklearn SelectFromModel
def test_sklearn_SelectFromModel(extracted_features_name='multiclass_180s_data.csv'):
    y=get_y()

    # 全部特征
    extracted_features = pd.read_csv(base_path+extracted_features_name)
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
    df.to_csv(base_path+'test_sklearn_SelectFromModel.csv')


# test sklearn ExtraTreesClassifier
def test_sklearn_ExtraTreesClassifier(extracted_features_name='multiclass_180s_data.csv'):
    y=get_y()

    # 全部特征
    extracted_features = pd.read_csv(base_path+extracted_features_name)
    # 除掉 id
    del extracted_features['id']
    # del extracted_features['Unnamed: 0']

    cols=extracted_features.columns.values.tolist()
    print('select start...')

    y = np.array(np.array(y).tolist())
    extracted_features_arr = np.array(extracted_features)
    print(extracted_features)
    print(y)
    clf = ExtraTreesClassifier(n_estimators=10, max_depth=4)
    clf = clf.fit(extracted_features_arr, y)
    res=SelectFromModel(clf, prefit=True)
    features_filtered = res.transform(extracted_features_arr)
    cols=get_cols(cols,res.get_support())
    print(np.array(features_filtered))


    df = pd.DataFrame(features_filtered, columns=cols)
    df.to_csv(base_path+'test_sklearn_ExtraTreesClassifier_4.csv')


# test
def get_cols(x,y):
    cols=[]
    for i in range(len(y)):
        if y[i]:
            cols.append(x[i])
    return cols


# test sklearn VarianceThreshold
def test_sklearn_VarianceThreshold(extracted_features_name='multiclass_180s_data.csv'):
    y=get_y()

    # 全部特征
    extracted_features = pd.read_csv(base_path+extracted_features_name)
    # 除掉 id
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
    df.to_csv(base_path+'test_sklearn_VarianceThreshold.csv')


def test_select_features_VarianceThreshold(extracted_features_name='test_sklearn_VarianceThreshold.csv'):
    y=get_y()

    # 全部特征
    extracted_features = pd.read_csv(base_path+extracted_features_name)
    del extracted_features['Unnamed: 0']

    print(extracted_features)
    print('select start...')

    # 选取较相关的特征
    # 可选属性 fdr_level = 0.05 ?
    features_filtered = select_features(extracted_features, y, n_jobs=1, fdr_level=0.01,ml_task='classification')
    print(features_filtered)
    features_filtered.to_csv(base_path+'select_features_VarianceThreshold.csv')
    print('select end')


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
