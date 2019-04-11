import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


def get_xy(name):
    # csv_data = pd.read_csv(name)
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]

    # if 'id' in csv_data.columns.values.tolist():
    #     del csv_data['id']
    # del csv_data['Unnamed: 0']

    pick = pd.read_csv('analyze_result_alpha1.csv')
    pick = list(np.array(pick['name'])[:10])
    print(pick)
    # csv_data=csv_data[pick]
    csv_data = pd.read_csv(name, usecols=pick)

    x = np.array(csv_data, dtype=float)

    return x, y


def naive_bayes_GaussianNB(x_train, x_test, y_train, y_test):
    # x_train = preprocessing.scale(x_train)
    # x_test = preprocessing.scale(x_test)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = GaussianNB()
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


# 决策树
def decide_tree(x_train, x_test, y_train, y_test):
    # x_train = preprocessing.scale(x_train)
    # x_test = preprocessing.scale(x_test)

    print(x_train)
    print(len(x_train[0]))
    print(x_test)
    print(len(x_test[0]))

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = tree.DecisionTreeClassifier(class_weight='balanced')
    clf = clf.fit(x_train, y_train.ravel())

    expected = y_test
    predicted = clf.predict(x_test)

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


def linear_svm(x_train, x_test, y_train, y_test):
    # x_train=preprocessing.scale(x_train)
    # x_test=preprocessing.scale(x_test)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = svm.SVC(kernel='linear', class_weight='balanced')
    clf.fit(x_train, y_train)

    # expected = y_train
    # predicted = clf.predict(x_train)
    # print(metrics.classification_report(expected, predicted))

    expected = y_test
    predicted = clf.predict(x_test)

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


def k_n_n(x_train, x_test, y_train, y_test):
    # x_train = preprocessing.scale(x_train)
    # x_test = preprocessing.scale(x_test)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)

    # expected = y_train
    # predicted = clf.predict(x_train)
    # print(metrics.classification_report(expected, predicted))

    expected = y_test
    predicted = clf.predict(x_test)

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


# 随机森林
def random_forest(x_train, x_test, y_train, y_test):
    # x_train = preprocessing.scale(x_train)
    # x_test = preprocessing.scale(x_test)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced')
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


def k_cv_3(name):
    colums = ['svm_precision', 'svm_recall', 'svm_accuracy', 'knn_precision', 'knn_recall', 'knn_accuracy',
              'tree_precision', 'tree_recall', 'tree_accuracy',
              'bayes_precision', 'bayes_recall', 'bayes_accuracy', 'forest_precision', 'forest_recall',
              'forest_accuracy']
    # colums = ['tree_precision', 'tree_recall', 'tree_accuracy',
    #           'bayes_precision', 'bayes_recall', 'bayes_accuracy', 'forest_precision', 'forest_recall',
    #           'forest_accuracy']
    acc_pd = pd.DataFrame(columns=colums)

    x, y = get_xy(name)

    kf = KFold(n_splits=6, shuffle=True)
    for train_index, test_index in kf.split(x):
        print(len(train_index))
        print(len(test_index))
        print('train_index', train_index, 'test_index', test_index)
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        temp = []

        # print('svm:')
        # precision, recall, accuracy = linear_svm(x_train, x_test, y_train, y_test)
        # temp.append(precision)
        # temp.append(recall)
        # temp.append(accuracy)
        # print(precision, recall, accuracy)

        # print('svm:')
        # precision, recall, accuracy = svm_train(x_train, x_test, y_train, y_test)
        # temp.append(precision)
        # temp.append(recall)
        # temp.append(accuracy)
        # print(precision, recall, accuracy)

        print('svm:')
        precision, recall, accuracy = linear_svm(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)
        print(precision, recall, accuracy)

        print('knn:')
        precision, recall, accuracy = k_n_n(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)
        print(precision, recall, accuracy)

        print('decide tree:')
        precision, recall, accuracy = decide_tree(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        print('native bayes:')
        precision, recall, accuracy = naive_bayes_GaussianNB(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        print('random forest:')
        precision, recall, accuracy = random_forest(x_train, x_test, y_train, y_test)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)

        acc_pd.loc[len(acc_pd)] = temp

    acc_pd.loc['mean'] = acc_pd.mean()
    acc_pd.to_csv(name[:-4] + '_classify_c_k_p_10.csv')


def start():
    file_names = ['select_features_VarianceThreshold.csv']
    for x in file_names:
        k_cv_3(x)
