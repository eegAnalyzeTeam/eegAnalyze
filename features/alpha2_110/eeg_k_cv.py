import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


lack_alpha1=[8,15,28]
lack_alpha2=[11,21,28,53,93,95]

def get_xy(name):
    csv_data = pd.read_csv(name)
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]

    y = np.delete(y, lack_alpha2, axis=0)

    # del csv_data['id']
    if 'id' in csv_data.columns.values.tolist():
        del csv_data['id']
    del csv_data['Unnamed: 0']
    x = np.array(csv_data, dtype=float)

    return x, y


# simple print result
def simple_k_cv():
    x,y=get_xy()
    clf = GaussianNB()
    scores = cross_val_score(clf, x, y, cv=10)
    print('bayes:')
    print(scores)
    print(scores.mean())

    clf = tree.DecisionTreeClassifier(class_weight='balanced')
    scores = cross_val_score(clf, x, y, cv=10)
    print('DecisionTree:')
    print(scores)
    print(scores.mean())

    clf = svm.SVC(kernel='linear',class_weight='balanced')
    scores = cross_val_score(clf, x, y, cv=10)
    print('svm:')
    print(scores)
    print(scores.mean())

    clf = RandomForestClassifier(n_estimators=100, max_depth=4,class_weight='balanced')
    scores = cross_val_score(clf, x, y, cv=10)
    print('random forest:')
    print(scores)
    print(scores.mean())


def naive_bayes_GaussianNB(x_train, x_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))  # 输出分类信息
    label = list(set(y_train))  # 去重复，得到标签类别
    matrix = metrics.confusion_matrix(expected, predicted, labels=label)
    print(matrix)  # 输出混淆矩阵信息

    accuracy_data=np.array(matrix)
    accuracy_little=accuracy_data[0][0]/(accuracy_data[0][0]+accuracy_data[0][1])
    accuracy_big=accuracy_data[1][1]/(accuracy_data[1][0]+accuracy_data[1][1])
    print(accuracy_little,accuracy_big)
    return clf.score(x_test, y_test),accuracy_little,accuracy_big


# 决策树
def decide_tree(x_train, x_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier(class_weight='balanced')
    clf = clf.fit(x_train, y_train.ravel())

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))  # 输出分类信息
    label = list(set(y_train))  # 去重复，得到标签类别
    matrix = metrics.confusion_matrix(expected, predicted, labels=label)
    print(matrix)  # 输出混淆矩阵信息

    accuracy_data=np.array(matrix)
    accuracy_little=accuracy_data[0][0]/(accuracy_data[0][0]+accuracy_data[0][1])
    accuracy_big=accuracy_data[1][1]/(accuracy_data[1][0]+accuracy_data[1][1])
    print(accuracy_little,accuracy_big)
    return clf.score(x_test, y_test),accuracy_little,accuracy_big


# svm
def svm_train(x_train, x_test, y_train, y_test):
    clf = svm.SVC(kernel='linear',class_weight='balanced')
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

    accuracy_data=np.array(matrix)
    accuracy_little=accuracy_data[0][0]/(accuracy_data[0][0]+accuracy_data[0][1])
    accuracy_big=accuracy_data[1][1]/(accuracy_data[1][0]+accuracy_data[1][1])
    print(accuracy_little,accuracy_big)
    return clf.score(x_test, y_test),accuracy_little,accuracy_big


# 随机森林
def random_forest(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, max_depth=4,class_weight='balanced')
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)
    print(metrics.classification_report(expected, predicted))  # 输出分类信息
    label = list(set(y_train))  # 去重复，得到标签类别
    matrix = metrics.confusion_matrix(expected, predicted, labels=label)
    print(matrix)  # 输出混淆矩阵信息

    accuracy_data=np.array(matrix)
    accuracy_little=accuracy_data[0][0]/(accuracy_data[0][0]+accuracy_data[0][1])
    accuracy_big=accuracy_data[1][1]/(accuracy_data[1][0]+accuracy_data[1][1])
    print(accuracy_little,accuracy_big)
    return clf.score(x_test, y_test),accuracy_little,accuracy_big


def k_cv(name):
    colums = ['svm', 'decide_tree', 'naive_bayes_GaussianNB', 'random_forest']
    res_pd = pd.DataFrame(columns=colums)

    colums = ['svm_less', 'svm_many', 'tree_less', 'tree_many', 'bayes_less', 'bayes_many', 'forest_less',
              'forest_many']
    acc_pd = pd.DataFrame(columns=colums)

    x,y=get_xy(name)
    kf = KFold(n_splits=6,shuffle=True)
    for train_index, test_index in kf.split(x):
        print('train_index', train_index, 'test_index', test_index)
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        temp = []
        acc_temp = []

        print('svm:')
        score_svm, little, big = svm_train(x_train, x_test, y_train, y_test)
        temp.append(score_svm)
        acc_temp.append(little)
        acc_temp.append(big)

        print('decide tree:')
        score_tree, little, big = decide_tree(x_train, x_test, y_train, y_test)
        temp.append(score_tree)
        acc_temp.append(little)
        acc_temp.append(big)

        print('native bayes:')
        score_bayes, little, big = naive_bayes_GaussianNB(x_train, x_test, y_train, y_test)
        temp.append(score_bayes)
        acc_temp.append(little)
        acc_temp.append(big)

        print('random forest:')
        score_forest, little, big = random_forest(x_train, x_test, y_train, y_test)
        temp.append(score_forest)
        acc_temp.append(little)
        acc_temp.append(big)

        res_pd.loc[len(res_pd)] = temp
        acc_pd.loc[len(res_pd)] = acc_temp

    res_pd.loc['mean'] = res_pd.mean()
    res_pd.to_csv(name[:-4]+'eeg_classify_kcv.csv')

    acc_pd.loc['mean'] = acc_pd.mean()
    acc_pd.to_csv(name[:-4]+'eeg_classify_acc_kcv.csv')


file_names=['tsfresh_filteredFeatures.csv','test_sklearn_SelectFromModel.csv','test_sklearn_ExtraTreesClassifier.csv','select_features_VarianceThreshold.csv']
for x in file_names:
    k_cv(x)
