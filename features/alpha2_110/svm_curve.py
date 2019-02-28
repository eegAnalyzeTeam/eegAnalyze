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

lack_alpha1=[8,15,28]
lack_alpha2=[11,21,28,53,93,95]
lack_all=[0,15,77]

def split_data():
    # handle_data()
    csv_data = pd.read_csv('tsfresh_filteredFeatures.csv')
    y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
    y = np.array(y_csv_data)[:, 1]

    y=np.delete(y,lack_alpha2,axis=0)

    # del csv_data['id']
    if 'id' in csv_data.columns.values.tolist():
        del csv_data['id']
    del csv_data['Unnamed: 0']
    x = np.array(csv_data, dtype=float)
    print(x)
    print(len(x))

    return x, y


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
            clf.fit(x_train, y_train.ravel())
            temp_train.append(clf.score(x_train, y_train))
            temp_test.append(clf.score(x_test, y_test))
        train_score.append(temp_train)
        test_score.append(temp_test)
    return train_sizes,np.array(train_score),np.array(test_score)


def svm_curve(clf,name):
    x, y = split_data()
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
    plt.savefig(name+'.png')
    plt.close()


def my_validation_curve():
    x, y = split_data()
    clf = svm.SVC(kernel='linear', C=1.3, decision_function_shape='ovo')

    param_range = [0.1,0.2,0.5,0.8,1,2,3,4,5,6,7,8,9,10, 20]
    train_score, test_score = validation_curve(clf, x, y, param_name='C',
                                               param_range=param_range, cv=10, scoring='accuracy')
    train_score = np.mean(train_score, axis=1)
    test_score = np.mean(test_score, axis=1)
    plt.plot(param_range, train_score, 'o-', color='r', label='training')
    plt.plot(param_range, test_score, 'o-', color='g', label='testing')
    plt.legend(loc='best')
    plt.xlabel('number of tree')
    plt.ylabel('accuracy')
    plt.savefig('validation_curve.png')
    plt.show()

name =['svm','bayes','decision tree','random forest']
clf = svm.SVC(kernel='linear', class_weight='balanced')
svm_curve(clf,name[0])
clf = GaussianNB()
svm_curve(clf,name[1])
clf = tree.DecisionTreeClassifier(class_weight='balanced')
svm_curve(clf,name[2])
clf = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced')
svm_curve(clf,name[3])
# my_validation_curve()