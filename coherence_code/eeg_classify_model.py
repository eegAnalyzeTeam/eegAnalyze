import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


def get_xy(name='coh_train.csv'):
    csv_data = pd.read_csv(name)
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]

    # del csv_data['id']
    del csv_data['Unnamed: 0']
    x = np.array(csv_data, dtype=float)

    return x, y


def split_data(i=1):
    x, y = get_xy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=i, train_size=0.7)
    return x_train, x_test, y_train.ravel(), y_test.ravel()


# 贝叶斯 高斯模型
def naive_bayes_GaussianNB(i=1):
    x_train, x_test, y_train, y_test = split_data(i)
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
def decide_tree(i=1):
    x_train, x_test, y_train, y_test = split_data(i)
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
def svm_train(i=1):
    x_train, x_test, y_train, y_test = split_data(i)
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
def random_forest(i=1):
    x_train, x_test, y_train, y_test = split_data(i)
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


def init_main():
    colums = ['svm', 'decide_tree', 'naive_bayes_GaussianNB', 'random_forest']
    res_pd = pd.DataFrame(columns=colums)

    colums = ['svm_less', 'svm_many', 'tree_less', 'tree_many', 'bayes_less', 'bayes_many', 'forest_less',
              'forest_many']
    acc_pd = pd.DataFrame(columns=colums)

    for i in range(30):
        temp = []
        acc_temp = []

        print('svm:')
        score_svm, little, big = svm_train(i)
        temp.append(score_svm)
        acc_temp.append(little)
        acc_temp.append(big)

        print('decide tree:')
        score_tree, little, big = decide_tree(i)
        temp.append(score_tree)
        acc_temp.append(little)
        acc_temp.append(big)

        print('native bayes:')
        score_bayes, little, big = naive_bayes_GaussianNB(i)
        temp.append(score_bayes)
        acc_temp.append(little)
        acc_temp.append(big)

        print('random forest:')
        score_forest, little, big = random_forest(i)
        temp.append(score_forest)
        acc_temp.append(little)
        acc_temp.append(big)

        res_pd.loc[len(res_pd)] = temp
        acc_pd.loc[len(res_pd)] = acc_temp
    res_pd.loc['mean'] = res_pd.mean()
    res_pd.to_csv('eeg_classify_models.csv')

    acc_pd.loc['mean'] = acc_pd.mean()
    acc_pd.to_csv('eeg_classify_acc.csv')

