import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import tree, metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

base_path = '/home/rbai/Multiclass/'


# 从文件中读取x、y
def get_xy(name):
    csv_data = pd.read_csv(base_path + name)
    y_csv_data = np.loadtxt(base_path + 'multiclass_60s_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)

    if 'id' in csv_data.columns.values.tolist():
        del csv_data['id']
    del csv_data['Unnamed: 0']
    x = np.array(csv_data, dtype=float)

    return x, y


# 划分训练集和测试集
def split_data(name, i=1):
    x, y = get_xy(name)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=i, train_size=0.8)
    return x_train, x_test, y_train.ravel(), y_test.ravel()


# 线性svm
def linear_svm_classify(name, i):
    x_train, x_test, y_train, y_test = split_data(name, i)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = svm.LinearSVC(penalty='l2', class_weight='balanced', loss='hinge')
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)

    print(metrics.classification_report(expected, predicted))  # 输出分类信息
    label = list(set(y_train))  # 去重复，得到标签类别
    matrix = metrics.confusion_matrix(expected, predicted, labels=label)
    print(matrix)  # 输出混淆矩阵信息

    return calcAccuracy(matrix)


# knn
def knn_classify(name, i):
    x_train, x_test, y_train, y_test = split_data(name, i)

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

    return calcAccuracy(matrix)


# 贝叶斯
def bayes_classify(name, i):
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

    return calcAccuracy(matrix)


# 决策树
def decide_tree_classify(name, i):
    x_train, x_test, y_train, y_test = split_data(name, i)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = tree.DecisionTreeClassifier(class_weight='balanced')
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)

    print(metrics.classification_report(expected, predicted))  # 输出分类信息
    label = list(set(y_train))  # 去重复，得到标签类别
    matrix = metrics.confusion_matrix(expected, predicted, labels=label)
    print(matrix)  # 输出混淆矩阵信息

    return calcAccuracy(matrix)


# 随机森林
def random_tree_classify(name, i):
    x_train, x_test, y_train, y_test = split_data(name, i)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
    x_train = scaling.transform(x_train)
    x_test = scaling.transform(x_test)

    clf = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced')
    clf.fit(x_train, y_train)

    expected = y_test
    predicted = clf.predict(x_test)

    print(metrics.classification_report(expected, predicted))  # 输出分类信息
    label = list(set(y_train))  # 去重复，得到标签类别
    matrix = metrics.confusion_matrix(expected, predicted, labels=label)
    print(matrix)  # 输出混淆矩阵信息

    return calcAccuracy(matrix)


# 根据混淆矩阵计算各个类别准确率
def calcAccuracy(matrix):
    data = np.array(matrix)
    res = []
    for i in range(len(data)):
        _sum = 0
        for j in range(len(data)):
            _sum += data[i][j]
        try:
            res.append(data[i][i] / _sum)
        except Exception:
            res.append(-1)

    return res


# 入口主函数
def start(name='select_features_VarianceThreshold.csv'):
    cols = ['svm_0', 'svm_1', 'svm_2', 'svm_3', 'svm_4',
            'knn_0', 'knn_1', 'knn_2', 'knn_3', 'knn_4',
            'bayes_0', 'bayes_1', 'bayes_2', 'bayes_3', 'bayes_4',
            'tree_0', 'tree_1', 'tree_2', 'tree_3', 'tree_4',
            'forests_0', 'forests_1', 'forests_2', 'forests_3', 'forests_4']
    df = pd.DataFrame(columns=cols)

    for i in range(20):
        res = linear_svm_classify(base_path + name, i)
        res += knn_classify(base_path + name, i)
        res += bayes_classify(base_path + name, i)
        res += decide_tree_classify(base_path + name, i)
        res += random_tree_classify(base_path + name, i)
        print(res)
        print(len(res))
        df.loc[len(df)] = res

    df.loc['mean'] = df.mean()
    df.to_csv(base_path + name[:-4] + 'test_.csv')
