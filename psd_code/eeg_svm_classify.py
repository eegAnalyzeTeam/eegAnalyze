import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from psd_code import eeg_psd_channel
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import matplotlib.pyplot as plt

N = 62


def get_pick(picks, all_name):
    res = []
    for x in all_name:
        if x not in picks:
            res.append(x)
    return res


def load_data():
    control_data = pd.read_csv('psd_c_alpha2.csv')
    patient_data = pd.read_csv('psd_p_alpha2.csv')
    return control_data, patient_data


def handle_data():
    control_data, patient_data = load_data()

    picks, all_name = eeg_psd_channel.pick_channel()
    global N
    pick = get_pick(picks, all_name)
    N = N - len(pick)

    control_data.drop(columns=['id', 'groupId', 'Unnamed: 0', 'average'], axis=1, inplace=True)
    control_data.drop(columns=pick, axis=1, inplace=True)
    control_data['type'] = 0.0
    patient_data.drop(columns=['id', 'groupId', 'Unnamed: 0', 'average'], axis=1, inplace=True)
    patient_data.drop(columns=pick, axis=1, inplace=True)
    patient_data['type'] = 1.0
    df = pd.merge(control_data, patient_data, how='outer')
    df.to_csv('format_psd_svm.csv', index=None)


def split_data(i):
    # handle_data()
    data = np.loadtxt('format_psd_svm.csv', dtype=float, skiprows=1, delimiter=',')
    msg = []
    global N
    if N == 62:
        picks, all_name = eeg_psd_channel.pick_channel()
        pick = get_pick(picks, all_name)
        N = N - len(pick)
    # N 待定
    x, y = np.split(data, (N,), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=i, train_size=0.7)
    msg.append("y_train_" + str(i) + ':' + str(np.sum(y_train.ravel())))
    return x_train, x_test, y_train, y_test, msg


# def get_same_data():
#         # handle_data()
#         data = np.loadtxt('format_psd_svm.csv', dtype=float, skiprows=1, delimiter=',')
#
#         global N
#         if N == 62:
#             picks, all_name = eeg_psd_channel.pick_channel()
#             pick = get_pick(picks, all_name)
#             N = N - len(pick)
#         # N 待定
#         x, y = np.split(data, (N,), axis=1)
#         x_train=x[:20]
#         x_train+=x[-20:]
#         y_train=y[:20]
#         y_train+=y[-20:]
#         x_test=x[20:-20]
#         y_test=y[20:-20]
#         print(x_train,y_train)
#         return x_train, x_test, y_train, y_test

def svm_train():
    out = []
    msgs = []
    for i in range(1, 15):
        x_train, x_test, y_train, y_test, msg = split_data(i)
        msgs.append(msg)
        clf = svm.SVC(C=1, kernel='rbf', gamma=20, decision_function_shape='ovo')
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
        out.append(str(i) + ':' + str(clf.score(x_test, y_test)))
        y_hat = clf.predict(x_test)
        print(classification_report(y_test, y_hat))

    print(out)
    print(np.array(msgs))


def start():
    handle_data()
    svm_train()
