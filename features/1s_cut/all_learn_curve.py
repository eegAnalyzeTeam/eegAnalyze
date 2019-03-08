from sklearn import svm
from sklearn.model_selection import learning_curve,validation_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# 读取需要的csv，并分割成测试集和训练集
def split_data(my_csv):
    # handle_data()
    csv_data = pd.read_csv(my_csv)
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]


    if 'id' in csv_data.columns.values.tolist():
        del csv_data['id']
    del csv_data['Unnamed: 0']
    x = np.array(csv_data, dtype=float)
    print(x)
    print(len(x))

    return x, y

# 根据传入的参数计算出模型的得分
def get_score(x,y,clf):
    train_sizes=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    train_score=[]
    test_score=[]
    # clf = svm.SVC(kernel='linear', C=1.3, decision_function_shape='ovr',class_weight='balanced')
    for i in train_sizes:
        temp_train=[]
        temp_test=[]
        for my_random in range(20):
            x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=my_random, train_size=i)

            scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
            x_train = scaling.transform(x_train)
            x_test = scaling.transform(x_test)

            clf.fit(x_train, y_train.ravel())
            temp_train.append(clf.score(x_train, y_train))
            temp_test.append(clf.score(x_test, y_test))
        train_score.append(temp_train)
        test_score.append(temp_test)
    return train_sizes,np.array(train_score),np.array(test_score)


# 根据传入的参数，画出学习曲线
def learn_curve(clf, name,my_csv):
    x, y = split_data(my_csv)
    # clf = svm.SVC(kernel='linear', C=1.3, decision_function_shape='ovo')
    print(y)
    print(len(y))
    train_sizes, train_score, test_score = get_score(x, y,clf)
    # train_sizes, train_score, test_score = learning_curve(clf,x,y,train_sizes=[0.1,0.2,0.4,0.6,0.8],cv=None,scoring='accuracy')
    # train_error = 1 - np.mean(train_score, axis=1)
    # test_error = 1 - np.mean(test_score, axis=1)
    train_score = np.mean(train_score, axis=1)
    test_score = np.mean(test_score, axis=1)
    plt.plot(train_sizes, train_score, 'o-', color='r', label='training')
    plt.plot(train_sizes, test_score, 'o-', color='g', label='testing')
    plt.legend(loc='best')
    plt.xlabel('traing examples')
    plt.ylabel('accuracy')
    plt.title(name +'  learning curve')
    plt.show()
    plt.savefig(my_csv+'_'+name+'.png')
    plt.close()


# 针对几种不同的模型画学习曲线
def init(my_csv):
    name =['svm','knn','bayes','decision tree','random forest']
    # svm
    clf = svm.LinearSVC(penalty='l2',class_weight='balanced',loss='hinge')
    learn_curve(clf, name[0],my_csv)
    # knn
    clf = KNeighborsClassifier()
    learn_curve(clf, name[1], my_csv)
    # 贝叶斯
    clf = GaussianNB()
    learn_curve(clf, name[2],my_csv)
    # 决策树
    clf = tree.DecisionTreeClassifier(class_weight='balanced')
    learn_curve(clf, name[3],my_csv)
    # 随机森林
    clf = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced')
    learn_curve(clf, name[4],my_csv)

# 画学习曲线的入口函数
def start():
    file_names = ['test_sklearn_ExtraTreesClassifier_4.csv']
    # 这里的for循环是为了画多个学习曲线的时候用
    for x in file_names:
        init(x)

start()