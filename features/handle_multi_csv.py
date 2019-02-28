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


def read_alpha1():
    base = pd.read_csv('features_change/alpha1/control_data_' + str(0) + '.csv')
    for i in range(1,30):
        temp = pd.read_csv('features_change/alpha1/control_data_' + str(i) + '.csv')
        base=base.append(temp)
    for i in range(30,111):
        temp = pd.read_csv('features_change/alpha1/patient_data_' + str(i) + '.csv')
        base=base.append(temp)

    base.to_csv('features_change/alpha1/tsfresh_data.csv')


def read_all():
    base = pd.read_csv('features_change/all/control_data_' + str(0) + '.csv')
    for i in range(1,30):
        temp = pd.read_csv('features_change/all/control_data_' + str(i) + '.csv')
        base=base.append(temp)
    for i in range(30,111):
        temp = pd.read_csv('features_change/all/patient_data_' + str(i) + '.csv')
        base=base.append(temp)

    base.to_csv('features_change/all/tsfresh_data.csv')



def init():
    read_alpha1()
    print('alpha1 end')
    read_alpha2()
    print('alpha2 end')
    read_all()
    print('all end')

init()