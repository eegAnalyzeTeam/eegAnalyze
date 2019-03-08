import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import math

# 将特征的列名，以及病人、正常人对应的这一列的数据抽取出来
def handle_data(name):
    df = pd.read_csv(name)

    del df['Unnamed: 0']

    colnums = df.columns.values.tolist()
    data = np.array(df)
    data_control = data[:5774, :]
    data_patient = data[5774:, :]

    return colnums, list(data_control.T), list(data_patient.T)


# 计算各个特征的p值，并且排序存成csv
def calculate_anova_p(name):
    colnums, control, patient = handle_data(name)

    res = {}
    for i in range(len(colnums)):
        f, p = f_oneway(control[i], patient[i])
        # 如果p值为nan则赋值为1
        if math.isnan(p):
            p=1
        res[colnums[i]] = p

    # 排序
    res=sorted(res.items(),key=lambda item:item[1])

    print(res)

    # 存储
    df =pd.DataFrame(columns=['name','p'])
    for x in res:
        temp=[]
        temp.append(x[0])
        temp.append(x[1])
        df.loc[len(df)]=x

    df.to_csv('analyze_result_all.csv')


calculate_anova_p('test_sklearn_ExtraTreesClassifier_4.csv')


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