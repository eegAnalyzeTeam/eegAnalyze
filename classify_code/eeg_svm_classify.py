import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import classification_report
import eeg_psd_channel
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


N=62

def get_pick(picks,all_name):
    res=[]
    for x in all_name:
        if x not in picks:
            res.append(x)
    return res


def load_data():
    control_data = pd.read_csv('psd_c_alpha1.csv')
    patient_data = pd.read_csv('psd_p_alpha1.csv')
    return control_data,patient_data


def handle_data():
    control_data, patient_data=load_data()

    picks,all_name=eeg_psd_channel.pick_channel()
    global N
    pick=get_pick(picks,all_name)
    N=N-len(pick)

    control_data.drop(columns=['id', 'groupId','Unnamed: 0','average'],axis=1,inplace=True)
    control_data.drop(columns=pick, axis=1, inplace=True)
    control_data['type'] = 0.0
    patient_data.drop(columns=['id', 'groupId', 'Unnamed: 0','average'],axis=1,inplace=True)
    patient_data.drop(columns=pick, axis=1, inplace=True)
    patient_data['type'] = 1.0
    df = pd.merge(control_data, patient_data, how='outer')

    df.to_csv('format_psd_svm.csv',index=None)


def split_data():
    handle_data()
    data = np.loadtxt('format_psd_svm.csv', dtype=float, skiprows=1,delimiter=',')

    # global N
    # if N==62:
    #     picks, all_name = eeg_psd_channel.pick_channel()
    #     pick = get_pick(picks, all_name)
    #     N = N - len(pick)

    print(len(data))
    print(data)
    # N 待定
    x, y = np.split(data, (N,), axis=1)
    # print(y)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)

    return x, y

def svm_train():
    x, y=split_data()
    y=y.ravel()
    clf = svm.SVC(C=6, kernel='rbf', gamma=20, decision_function_shape='ovr')
    print(x)
    print(y)
    # clf.fit(x_train, y_train.ravel())

    # joblib.dump(clf, 'clf.model')  # 保存模型

    # print('训练集:')
    # print(clf.score(x_train, y_train))
    # y_hat = clf.predict(x_train)
    # print(classification_report(y_train, y_hat))
    #
    #
    # print('测试集:')
    # print(clf.score(x_test, y_test))
    # y_hat = clf.predict(x_test)
    # print(classification_report(y_test, y_hat))

    scores = cross_val_score(clf, x, y, cv=8)  # 8折交叉验证
    print(scores)
    print(np.mean(scores))


def LogisticRegression_train():
    x, y=split_data()
    y=y.ravel()
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    print(x)
    print(y)
    # clf.fit(x_train, y_train.ravel())

    # joblib.dump(clf, 'clf.model')  # 保存模型

    # print('训练集:')
    # print(clf.score(x_train, y_train))
    # y_hat = clf.predict(x_train)
    # print(classification_report(y_train, y_hat))
    #
    #
    # print('测试集:')
    # print(clf.score(x_test, y_test))
    # y_hat = clf.predict(x_test)
    # print(classification_report(y_test, y_hat))

    scores = cross_val_score(clf, x, y, cv=8)  # 8折交叉验证
    print(scores)
    print(np.mean(scores))


LogisticRegression_train()