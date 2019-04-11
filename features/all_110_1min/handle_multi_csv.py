import pandas as pd


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


def start():
    read_allcut_extracedFeatures()
