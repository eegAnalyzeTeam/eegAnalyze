import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib


def get_xy(name):
    csv_data = pd.read_csv(name)
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]

    if 'id' in csv_data.columns.values.tolist():
        del csv_data['id']
    del csv_data['Unnamed: 0']
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
def decide_tree(x_train, x_test, y_train, y_test, i):
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

    if i == 0:
        joblib.dump(clf, "tree_model.m")
        joblib.dump(scaling, "tree_scaling.m")

    expected = y_test
    predicted = clf.predict(x_test)

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


def linear_svm(x_train, x_test, y_train, y_test, i):
    # x_train=preprocessing.scale(x_train)
    # x_test=preprocessing.scale(x_test)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = svm.LinearSVC(penalty='l2', class_weight='balanced', loss='hinge')
    clf.fit(x_train, y_train)

    if i == 0:
        joblib.dump(clf, "svm_model.m")
        joblib.dump(scaling, "svm_scaling.m")
    # expected = y_train
    # predicted = clf.predict(x_train)
    # print(metrics.classification_report(expected, predicted))

    expected = y_test
    predicted = clf.predict(x_test)

    return precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted)


def k_n_n(x_train, x_test, y_train, y_test, i):
    # x_train = preprocessing.scale(x_train)
    # x_test = preprocessing.scale(x_test)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = KNeighborsClassifier()
    clf.fit(x_train, y_train)

    if i == 5:
        joblib.dump(clf, "knn_model.m")
        joblib.dump(scaling, "knn_scaling.m")
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

    # x, y = get_xy(name)

    x, x_test, y, y_test = get_train_test(name)

    pd.DataFrame(x).to_csv('x_test.csv', header=False, index=False)
    pd.DataFrame(y).to_csv('y_test.csv', header=False, index=False)

    kf = KFold(n_splits=10, shuffle=True)
    i = 0
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
        precision, recall, accuracy = linear_svm(x_train, x_test, y_train, y_test, i)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)
        print(precision, recall, accuracy)

        print('knn:')
        precision, recall, accuracy = k_n_n(x_train, x_test, y_train, y_test, i)
        temp.append(precision)
        temp.append(recall)
        temp.append(accuracy)
        print(precision, recall, accuracy)

        print('decide tree:')
        precision, recall, accuracy = decide_tree(x_train, x_test, y_train, y_test, i)
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
        i += 1

    acc_pd.loc['mean'] = acc_pd.mean()
    acc_pd.to_csv(name[:-4] + '_c_k_clf_test.csv')


def get_train_test(name):
    csv_data = pd.read_csv(name)
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]

    if 'id' in csv_data.columns.values.tolist():
        del csv_data['id']
    del csv_data['Unnamed: 0']
    x = np.array(csv_data, dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)

    return x_train, x_test, y_train.ravel(), y_test.ravel()


def get_3test(name_x, name_y):
    x_test = np.loadtxt(name_x, delimiter=",")
    y_test = np.loadtxt(name_y, delimiter=",")

    print('tree:')
    scaling = joblib.load("tree_scaling.m")
    x_test = scaling.transform(x_test)
    clf = joblib.load("tree_model.m")
    expected = y_test.ravel()
    predicted = clf.predict(x_test)
    print(precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted))

    x_test = np.loadtxt(name_x, delimiter=",")
    y_test = np.loadtxt(name_y, delimiter=",")

    print('svm:')
    scaling = joblib.load("svm_scaling.m")
    x_test = scaling.transform(x_test)
    clf = joblib.load("svm_model.m")
    expected = y_test.ravel()
    predicted = clf.predict(x_test)
    print(precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted))

    x_test = np.loadtxt(name_x, delimiter=",")
    y_test = np.loadtxt(name_y, delimiter=",")

    print('knn:')
    scaling = joblib.load("knn_scaling.m")
    x_test = scaling.transform(x_test)
    clf = joblib.load("knn_model.m")
    expected = y_test.ravel()
    predicted = clf.predict(x_test)
    print(precision_score(expected, predicted), recall_score(expected, predicted), accuracy_score(expected, predicted))


def start():
    file_names = ['test_sklearn_ExtraTreesClassifier_8.csv']
    for x in file_names:
        k_cv_3(x)

    get_3test('x_test.csv', 'y_test.csv')
