import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


# 贝叶斯 高斯模型
def naive_bayes_GaussianNB(x_train, x_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))  # 输出分类信息
    label = list(set(y_train))  # 去重复，得到标签类别
    matrix = metrics.confusion_matrix(expected, predicted, labels=label)
    print(matrix)  # 输出混淆矩阵信息

    accuracy_data = np.array(matrix)
    accuracy_little = accuracy_data[0][0] / (accuracy_data[0][0] + accuracy_data[0][1])
    accuracy_big = accuracy_data[1][1] / (accuracy_data[1][0] + accuracy_data[1][1])
    print(accuracy_little, accuracy_big)
    return clf.score(x_test, y_test), accuracy_little, accuracy_big


# 决策树
def decide_tree(x_train, x_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier(criterion='gini', class_weight='balanced')
    # clf = tree.DecisionTreeClassifier(criterion='entropy',class_weight='balanced')

    clf = clf.fit(x_train, y_train.ravel())

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))  # 输出分类信息
    label = list(set(y_train))  # 去重复，得到标签类别
    matrix = metrics.confusion_matrix(expected, predicted, labels=label)
    print(matrix)  # 输出混淆矩阵信息

    accuracy_data = np.array(matrix)
    accuracy_little = accuracy_data[0][0] / (accuracy_data[0][0] + accuracy_data[0][1])
    accuracy_big = accuracy_data[1][1] / (accuracy_data[1][0] + accuracy_data[1][1])
    print(accuracy_little, accuracy_big)
    return clf.score(x_test, y_test), accuracy_little, accuracy_big


# svm
def svm_train(x_train, x_test, y_train, y_test):
    clf = svm.SVC(C=1, kernel='linear', class_weight='balanced')
    # clf = svm.SVC(C=10,kernel='rbf',class_weight='balanced')

    clf.fit(x_train, y_train)

    expected = y_train
    predicted = clf.predict(x_train)
    print(metrics.classification_report(expected, predicted))

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))  # 输出分类信息
    label = list(set(y_train))  # 去重复，得到标签类别
    matrix = metrics.confusion_matrix(expected, predicted, labels=label)
    print(matrix)  # 输出混淆矩阵信息

    accuracy_data = np.array(matrix)
    accuracy_little = accuracy_data[0][0] / (accuracy_data[0][0] + accuracy_data[0][1])
    accuracy_big = accuracy_data[1][1] / (accuracy_data[1][0] + accuracy_data[1][1])
    print(accuracy_little, accuracy_big)
    return clf.score(x_test, y_test), accuracy_little, accuracy_big


# 随机森林
def random_forest(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced')
    # clf = RandomForestClassifier(n_estimators=200, max_depth=5,class_weight='balanced')

    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))  # 输出分类信息
    label = list(set(y_train))  # 去重复，得到标签类别
    matrix = metrics.confusion_matrix(expected, predicted, labels=label)
    print(matrix)  # 输出混淆矩阵信息

    accuracy_data = np.array(matrix)
    accuracy_little = accuracy_data[0][0] / (accuracy_data[0][0] + accuracy_data[0][1])
    accuracy_big = accuracy_data[1][1] / (accuracy_data[1][0] + accuracy_data[1][1])
    print(accuracy_little, accuracy_big)
    return clf.score(x_test, y_test), accuracy_little, accuracy_big


