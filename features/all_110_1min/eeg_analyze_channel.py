import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import math


def handle_data(name):
    df = pd.read_csv(name)

    del df['Unnamed: 0']

    colnums = df.columns.values.tolist()
    data = np.array(df)
    data_control = data[:28, :]
    data_patient = data[28:, :]

    return colnums, list(data_control.T), list(data_patient.T)


def calculate_anova_p(name):
    colnums, control, patient = handle_data(name)

    res = {}
    for i in range(len(colnums)):
        f, p = f_oneway(control[i], patient[i])
        if math.isnan(p):
            p = 1
        res[colnums[i]] = p

    res = sorted(res.items(), key=lambda item: item[1])

    print(res)

    df = pd.DataFrame(columns=['name', 'p'])
    for x in res:
        temp = []
        temp.append(x[0])
        temp.append(x[1])
        df.loc[len(df)] = x

    df.to_csv('analyze_result.csv')


def start():
    calculate_anova_p('select_features_VarianceThreshold.csv')
