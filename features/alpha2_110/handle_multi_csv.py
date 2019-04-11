import numpy as np
import pandas as pd


def read_alpha2():
    base = pd.read_csv('features_change/alpha2/control_data_' + str(0) + '.csv')
    for i in range(1,30):
        temp = pd.read_csv('features_change/alpha2/control_data_' + str(i) + '.csv')
        base=base.append(temp)
    for i in range(30,111):
        temp = pd.read_csv('features_change/alpha2/patient_data_' + str(i) + '.csv')
        base=base.append(temp)

    base.to_csv('features_change/alpha2/tsfresh_data.csv')

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
    read_alpha2()
    get_svm_y()
