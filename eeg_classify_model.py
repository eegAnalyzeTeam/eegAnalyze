import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm


def get_xy():
    csv_data = pd.read_csv('test_sklearn_SelectFromModel.csv')
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]

    # del csv_data['id']
    del csv_data['Unnamed: 0']
    x = np.array(csv_data, dtype=float)
    print(x)
    print(y)

    return x, y


def split_data(i=1):
    x,y=get_xy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=i, train_size=0.6)
    return x_train, x_test, y_train.ravel(), y_test.ravel()


# 贝叶斯 高斯模型
def naive_bayes_GaussianNB():
    x_train, x_test, y_train, y_test = split_data()
    clf = GaussianNB()
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))  # 输出分类信息
    label = list(set(y_train))  # 去重复，得到标签类别
    print(metrics.confusion_matrix(expected, predicted, labels=label))  # 输出混淆矩阵信息


def decide_tree():
    x_train, x_test, y_train, y_test = split_data()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train.ravel())

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))  # 输出分类信息
    label = list(set(y_train))  # 去重复，得到标签类别
    print(metrics.confusion_matrix(expected, predicted, labels=label))  # 输出混淆矩阵信息


def svm_train():
    x_train, x_test, y_train, y_test = split_data()
    clf = svm.SVC(kernel='linear', C=1.3, decision_function_shape='ovo')
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))  # 输出分类信息
    label = list(set(y_train))  # 去重复，得到标签类别
    print(metrics.confusion_matrix(expected, predicted, labels=label))  # 输出混淆矩阵信息



# decide_tree()
# naive_bayes_GaussianNB()
svm_train()