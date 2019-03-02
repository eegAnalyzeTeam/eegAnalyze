import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def PolynomialSVC(degree, C=3):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        # ('linearSVC', svm.LinearSVC(C=C))#注意这两句都行
        ('kernelSVC', svm.SVC(kernel='linear', degree=degree, C=C))  # 注意这两句都行
    ])




def get_xy(name):
    csv_data = pd.read_csv(name)
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]

    # y = np.delete(y, lack_alpha1, axis=0)

    # del csv_data['id']
    if 'id' in csv_data.columns.values.tolist():
        del csv_data['id']
    del csv_data['Unnamed: 0']
    x = np.array(csv_data, dtype=float)


    return x, y


def split_data(name, i=1):
    x, y = get_xy(name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=i, train_size=0.5)
    return x_train, x_test, y_train.ravel(), y_test.ravel()


# 贝叶斯 高斯模型
def naive_bayes_GaussianNB(name, i=1):
    x_train, x_test, y_train, y_test = split_data(name, i)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

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
    # return  precision_score(expected,predicted),recall_score(expected,predicted),accuracy_score(expected,predicted)



def k_n_n(name, i=1):
    x_train, x_test, y_train, y_test = split_data(name, i)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)
    clf = KNeighborsClassifier()
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
    # return  precision_score(expected,predicted),recall_score(expected,predicted),accuracy_score(expected,predicted)


# 决策树
def decide_tree(name, i=1):
    x_train, x_test, y_train, y_test = split_data(name, i)
    # x_train=preprocessing.scale(x_train)
    #     # x_test=preprocessing.scale(x_test)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = tree.DecisionTreeClassifier(class_weight='balanced')
    # clf = tree.DecisionTreeClassifier(criterion='entropy',class_weight='balanced')

    clf.fit(x_train, y_train.ravel())

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

    # return  precision_score(expected,predicted),recall_score(expected,predicted),accuracy_score(expected,predicted)


# svm
def svm_train(name, i=1):
    x_train, x_test, y_train, y_test = split_data(name, i)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    x_train = preprocessing.scale(x_train)
    x_test = preprocessing.scale(x_test)
    clf = svm.LinearSVC(penalty='l2',class_weight='balanced',loss='hinge')
    # clf = svm.SVC(C=10,kernel='rbf',class_weight='balanced')

    clf.fit(x_train, y_train.ravel())
    print('fit success')

    # expected = y_train
    # predicted = clf.predict(x_train)
    # print(metrics.classification_report(expected, predicted))

    expected = y_test
    predicted = clf.predict(x_test)
    # print(metrics.classification_report(expected, predicted))  # 输出分类信息
    # label = list(set(y_train))  # 去重复，得到标签类别
    # matrix = metrics.confusion_matrix(expected, predicted, labels=label)
    # print(matrix)  # 输出混淆矩阵信息
    #
    #
    # accuracy_data=np.array(matrix)
    # accuracy_little=accuracy_data[0][0]/(accuracy_data[0][0]+accuracy_data[0][1])
    # accuracy_big=accuracy_data[1][1]/(accuracy_data[1][0]+accuracy_data[1][1])
    # print(accuracy_little,accuracy_big)
    # return clf.score(x_test, y_test),accuracy_little,accuracy_big
    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)




def linear_svm(name, i=1):
    x_train, x_test, y_train, y_test = split_data(name, i)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)


    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = svm.LinearSVC(class_weight='balanced', loss='hinge')
    # clf = svm.SVC(C=10,kernel='rbf',class_weight='balanced')

    clf.fit(x_train, y_train.ravel())
    print('fit success')

    # expected = y_train
    # predicted = clf.predict(x_train)
    # print(metrics.classification_report(expected, predicted))

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
    # return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)






# 随机森林
def random_forest(name, i=1):
    x_train, x_test, y_train, y_test = split_data(name, i)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)
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
    # return  precision_score(expected,predicted),recall_score(expected,predicted),accuracy_score(expected,predicted)




def init_main(name):
    colums = ['svm','knn', 'decide_tree', 'naive_bayes_GaussianNB', 'random_forest']
    res_pd = pd.DataFrame(columns=colums)

    colums = ['svm_less', 'svm_many', 'knn_less', 'knn_many','tree_less', 'tree_many', 'bayes_less', 'bayes_many', 'forest_less',
              'forest_many']
    acc_pd = pd.DataFrame(columns=colums)

    for i in range(30):
        temp = []
        acc_temp = []

        # print('svm:')
        # score_svm,little,big=svm_train(name,i)
        # temp.append(score_svm)
        # acc_temp.append(little)
        # acc_temp.append(big)

        print('svm:')
        score_svm, little, big = linear_svm(name, i)
        temp.append(score_svm)
        acc_temp.append(little)
        acc_temp.append(big)

        print('knn:')
        score_tree, little, big = k_n_n(name, i)
        temp.append(score_tree)
        acc_temp.append(little)
        acc_temp.append(big)

        print('decide tree:')
        score_tree, little, big = decide_tree(name, i)
        temp.append(score_tree)
        acc_temp.append(little)
        acc_temp.append(big)

        print('native bayes:')
        score_bayes, little, big = naive_bayes_GaussianNB(name, i)
        temp.append(score_bayes)
        acc_temp.append(little)
        acc_temp.append(big)

        print('random forest:')
        score_forest, little, big = random_forest(name, i)
        temp.append(score_forest)
        acc_temp.append(little)
        acc_temp.append(big)

        res_pd.loc[len(res_pd)] = temp
        acc_pd.loc[len(res_pd)] = acc_temp
    res_pd.loc['mean'] = res_pd.mean()
    res_pd.to_csv(name[:-4] + '_classify_models.csv')

    acc_pd.loc['mean'] = acc_pd.mean()
    acc_pd.to_csv(name[:-4] + '_classify_acc.csv')


def init_main_3(name):
    colums = ['svm_precision', 'svm_recall', 'svm_accuracy', 'tree_precision', 'tree_recall', 'tree_accuracy',
              'bayes_precision', 'bayes_recall', 'bayes_accuracy', 'forest_precision', 'forest_recall',
              'forest_accuracy']
    acc_pd = pd.DataFrame(columns=colums)

    for i in range(30):
        temp = []

        # print('svm:')
        # precision,recall,accuracy=svm_train(name,i)
        # temp.append(precision)
        # temp.append(recall)
        # temp.append(accuracy)
        # print(precision,recall,accuracy)

        # print('svm:')
        # precision, recall, accuracy = nu_svm(name, i)
        # temp.append(precision)
        # temp.append(recall)
        # temp.append(accuracy)
        # print(precision, recall, accuracy)

        print('svm:')
        precision, recall, accuracy = linear_svm(name, i)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)
        print(precision, recall, accuracy)

        # print('svm:')
        # precision, recall, accuracy = SGDRegressor1(name, i)
        # temp.append(precision)
        # temp.append(recall)
        # temp.append(accuracy)
        # print(precision, recall, accuracy)

        # print('svm:')
        # precision, recall, accuracy = _PolynomialSVC(name, i)
        # temp.append(precision)
        # temp.append(recall)
        # temp.append(accuracy)
        # print(precision, recall, accuracy)

        print('decide tree:')
        precision, recall, accuracy = decide_tree(name, i)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        print('native bayes:')
        precision, recall, accuracy = naive_bayes_GaussianNB(name, i)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        print('random forest:')
        precision, recall, accuracy = random_forest(name, i)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        acc_pd.loc[len(acc_pd)] = temp

    acc_pd.loc['mean'] = acc_pd.mean()
    acc_pd.to_csv(name[:-4] + '_classify_1.csv')


init_main('test_sklearn_ExtraTreesClassifier.csv')
