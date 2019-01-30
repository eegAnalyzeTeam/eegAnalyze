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


def handle_y(y):
    y=y.drop_duplicates(subset=['id', 'y'], keep='first')
    y=y.reset_index(drop=True)
    y=y.iloc[:,-1]

    return y


# 有效特征
def get_features(file_name):
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
    extracted_features.to_csv('tsfresh_extractedFeatures.csv')
    print('all features end')
    # 选取较相关的特征
    # 可选属性 fdr_level = 0.05 ?
    features_filtered = select_features(extracted_features, y,ml_task='classification',n_jobs=1,fdr_level =0.05)
    features_filtered.to_csv('tsfresh_filteredFeatures.csv')


# 从文件读取feature,在已经保存全部特征的情况下使用
def _select_features(extracted_features_name='tsfresh_extractedFeatures.csv'):
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]

    print(y)

    # 全部特征
    extracted_features = pd.read_csv(extracted_features_name)
    print(extracted_features)
    print('select start...')

    # 选取较相关的特征
    # 可选属性 fdr_level = 0.05 ?
    features_filtered = select_features(extracted_features, y ,n_jobs=1,fdr_level =1,ml_task='classification')
    print(features_filtered)
    features_filtered.to_csv('tsfresh_filteredFeatures_1.csv')
    print('select end')


# def test_sklearn_SelectKBest():
#     csv_data = pd.read_csv('tsfresh_data.csv')
#     y = csv_data[['id', 'y']]
#     y = handle_y(y)
#
#     print(y)
#
#     # 全部特征
#     extracted_features = pd.read_csv('tsfresh_extractedFeatures.csv')
#     # 除掉 id
#     del extracted_features['id']
#     print('select start...')
#
#     y = np.array(np.array(y).tolist())
#     extracted_features_arr = np.array(extracted_features)
#     print(extracted_features)
#     print(y)
#     # features_filtered = SelectKBest(SelectFdr, k=20).fit_transform(extracted_features_arr, y)
#     features_filtered = SelectFdr(chi2, alpha=0.01).fit_transform(extracted_features_arr, y)
#     print(np.array(features_filtered))
#
#     # 获取列名？
#     res_col = []
#     arr = np.array(features_filtered).T
#     for i in arr:
#         for indexs in extracted_features.columns:
#             if list(extracted_features[indexs]) == list(i):
#                 res_col.append(indexs)
#                 break
#     df = pd.DataFrame(features_filtered, columns=res_col)
#     df.to_csv('test_sklearn_SelectKBest.csv')

# test sklearn SelectFromModel
def test_sklearn_SelectFromModel(extracted_features_name='tsfresh_extractedFeatures.csv'):
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]

    print(y)

    # 全部特征
    extracted_features = pd.read_csv(extracted_features_name)
    # 除掉 id
    del extracted_features['id']
    cols=extracted_features.columns.values.tolist()
    print('select start...')

    y = np.array(np.array(y).tolist())
    extracted_features_arr = np.array(extracted_features)
    print(extracted_features)
    print(y)
    lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(extracted_features_arr, y)
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


def split_data(i):
    # handle_data()
    csv_data = pd.read_csv('test_sklearn_SelectFromModel.csv')
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:,1]

    del csv_data['Unnamed: 0']
    x=np.array(csv_data,dtype=float)
    print(x)
    msg=[]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=i, train_size=0.6)
    msg.append("y_train_"+str(i)+':'+str(np.sum(y_train.ravel())))
    return x_train, x_test, y_train, y_test,msg


def svm_train():
    out=[]
    msgs=[]
    test_pre=[]
    for i in range(1, 21):
        x_train, x_test, y_train, y_test, msg = split_data(i)
        msgs.append(msg)
        clf = svm.SVC(kernel='linear',C=1.3, decision_function_shape='ovo')
        # scores = cross_val_score(clf, x_test, y_test, cv=10)
        # print(scores)
        # out.append(str(i)+':'+str(scores.mean()))
        # print(str(scores.mean()))
        clf.fit(x_train, y_train.ravel())

        joblib.dump(clf, 'clf.model')  # 保存模型

        print('训练集:')
        print(clf.score(x_train, y_train))
        y_hat = clf.predict(x_train)
        print(classification_report(y_train, y_hat))

        print('测试集:')
        print(clf.score(x_test, y_test))
        test_pre.append(clf.score(x_test, y_test))
        out.append(str(i) + ':' + str(clf.score(x_test, y_test)))
        y_hat = clf.predict(x_test)
        print(classification_report(y_test, y_hat))
    print(test_pre)

# test sklearn ExtraTreesClassifier
# not used
def test_sklearn_ExtraTreesClassifier(extracted_features_name='tsfresh_extractedFeatures.csv'):
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]

    print(y)

    # 全部特征
    extracted_features = pd.read_csv(extracted_features_name)
    # 除掉 id
    del extracted_features['id']
    cols=extracted_features.columns.values.tolist()
    print('select start...')

    y = np.array(np.array(y).tolist())
    extracted_features_arr = np.array(extracted_features)
    print(extracted_features)
    print(y)
    clf = ExtraTreesClassifier(n_estimators=50, max_depth=4)
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
    df.to_csv('test_sklearn_ExtraTreesClassifier.csv')

# test
def get_cols(x,y):
    cols=[]
    for i in range(len(y)):
        if y[i]:
            cols.append(x[i])
    return cols


# test sklearn VarianceThreshold
# not used
def test_sklearn_VarianceThreshold(extracted_features_name='tsfresh_extractedFeatures.csv'):
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]

    print(y)

    # 全部特征
    extracted_features = pd.read_csv(extracted_features_name)
    # 除掉 id
    del extracted_features['id']
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

# not used
def test_select_features_VarianceThreshold(extracted_features_name='test_sklearn_VarianceThreshold.csv'):
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]

    print(y)

    # 全部特征
    extracted_features = pd.read_csv(extracted_features_name)
    del extracted_features['Unnamed: 0']

    print(extracted_features)
    print('select start...')

    # 选取较相关的特征
    # 可选属性 fdr_level = 0.05 ?
    features_filtered = select_features(extracted_features, y, n_jobs=1, fdr_level=3.8,ml_task='classification')
    print(features_filtered)
    features_filtered.to_csv('select_features_VarianceThreshold.csv')
    print('select end')


def start(name):
    get_features(name)
    test_sklearn_SelectFromModel()
    test_sklearn_ExtraTreesClassifier()
    test_sklearn_VarianceThreshold()
    test_select_features_VarianceThreshold()

    _select_features()


# get_features('tsfresh_data_alpha2.csv')
# _select_features()
# test_sklearn_ExtraTreesClassifier()
# test_sklearn_SelectFromModel()
# test_sklearn_VarianceThreshold()
# test_select_features_VarianceThreshold()
# test_sklearn_SelectKBest()
# test_sklearn_SelectFromModel()
# svm_train()

start('tsfresh_data_alpha2.csv')