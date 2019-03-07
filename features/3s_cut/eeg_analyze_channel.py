import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


def handle_data(name):
    df = pd.read_csv(name)

    del df['Unnamed: 0']

    colnums = df.columns.values.tolist()
    data = np.array(df)
    data_control = data[:1913, :]
    data_patient = data[1913:, :]

    return colnums, list(data_control.T), list(data_patient.T)


def calculate_anova_p(name):
    colnums, control, patient = handle_data(name)

    res = {}
    for i in range(88):
        f, p = f_oneway(control[i], patient[i])
        res[colnums[i]] = p

    res=sorted(res.items(),key=lambda item:item[1])

    print(res)

    df =pd.DataFrame(columns=['name','p'])
    for x in res:
        temp=[]
        temp.append(x[0])
        temp.append(x[1])
        df.loc[len(df)]=x

    df.to_csv('analyze_result.csv')

calculate_anova_p('test_sklearn_ExtraTreesClassifier.csv')


# def get_data(name):
#     df = pd.read_csv(name)
#
#     del df['Unnamed: 0']
#
#     y_csv_data = np.loadtxt('svm_y.csv', dtype=float, delimiter=',')
#     y = np.array(y_csv_data)[:, 1]
#
#     df['type']=y
#
#     anova_res = anova_lm(ols('CP5__mean_abs_change ~ C(type)', df).fit(), typ=1)
#
#     print(anova_res)
# get_data('test_sklearn_ExtraTreesClassifier.csv')