import pandas as pd
import numpy as np

def read_allcut_extracedFeatures():
    lack_cutall=[0,15,77]
    base = pd.read_csv('tsfresh_extractedFeatures' + str(1) + '.csv')
    for i in range(2, 111):
        if i in lack_cutall:
            continue
        temp = pd.read_csv('tsfresh_extractedFeatures' + str(i) + '.csv')
        base = base.append(temp)
        print('all '+str(i))

    base.to_csv('tsfresh_extractedFeatures.csv')


# 根据顺序构造出每个人对应的y值
def get_svm_y():
    content = []

    for i in range(0, 30):
        content.append(0)

    for i in range(30, 111):
        content.append(1)

    content = np.array(content).T
    df = pd.DataFrame(content)
    df.to_csv('svm_y.csv', header=0)


def start():
    read_allcut_extracedFeatures()
    get_svm_y()
