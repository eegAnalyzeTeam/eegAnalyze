#
#   根据上一步计算的FDs，根据左额叶、右额叶、前额叶求平均后，进行anova分析
#
import numpy as np
from scipy.stats import f_oneway
import math
import const
import csv
import pandas as pd


# 根据额叶、左额叶、右额叶的电极求均值
def average_fds(band, kind):
    data = np.loadtxt('csv/' + band + '_' + kind + '.csv', delimiter=',', dtype=np.str)
    data = data[:, 1:]
    colmn = data[0]

    data = np.array(data[1:, :], dtype=np.float)

    # left
    kfd = None
    hfd = None
    for i in range(len(colmn)):
        for x in const.left_brain:
            # 判定是左额叶电极
            if colmn[i].find(x) >= 0:
                if colmn[i].find('HFD') >= 0:
                    if hfd is None:
                        hfd = data[:, i]
                    else:
                        hfd = np.column_stack((hfd, data[:, i]))
                if colmn[i].find('KFD') >= 0:
                    if kfd is None:
                        kfd = data[:, i]
                    else:
                        kfd = np.column_stack((kfd, data[:, i]))
    res = np.mean(kfd, axis=1)
    hfd = np.mean(hfd, axis=1)
    res = np.column_stack((res, hfd))

    # right
    kfd = None
    hfd = None
    for i in range(len(colmn)):
        for x in const.right_brain:
            # 判定是左额叶电极
            if colmn[i].find(x) >= 0:
                if colmn[i].find('HFD') >= 0:
                    if hfd is None:
                        hfd = data[:, i]
                    else:
                        hfd = np.column_stack((hfd, data[:, i]))
                if colmn[i].find('KFD') >= 0:
                    if kfd is None:
                        kfd = data[:, i]
                    else:
                        kfd = np.column_stack((kfd, data[:, i]))
    kfd = np.mean(kfd, axis=1)
    hfd = np.mean(hfd, axis=1)
    res = np.column_stack((res, kfd))
    res = np.column_stack((res, hfd))

    # all
    kfd = None
    hfd = None
    for i in range(len(colmn)):
        if colmn[i].find('HFD') >= 0:
            if hfd is None:
                hfd = data[:, i]
            else:
                hfd = np.column_stack((hfd, data[:, i]))
        if colmn[i].find('KFD') >= 0:
            if kfd is None:
                kfd = data[:, i]
            else:
                kfd = np.column_stack((kfd, data[:, i]))
    kfd = np.mean(kfd, axis=1)
    hfd = np.mean(hfd, axis=1)

    res = np.column_stack((res, kfd))
    res = np.column_stack((res, hfd))

    fileread = open('csv/' + band + '_' + kind + '_average.csv', 'w', newline='')
    writer = csv.writer(fileread)
    title = ['left_kfd', 'left_hfd', 'right_kfd', 'right_hfd', 'all_kfd', 'all_hfd']
    writer.writerow(title)
    writer.writerows(res)
    fileread.close()


def get_csv():
    for band in const.bands_name:
        for kind in ['patient', 'control']:
            average_fds(band, kind)


def average_list(band, kind):
    data = np.loadtxt('csv/' + band + '_' + kind + '_average.csv', delimiter=',', dtype=np.str)
    colmn = data[0]
    data = np.array(data[1:, :], dtype=np.float).T
    res = {}
    for i in range(len(colmn)):
        res[colmn[i]] = data[i]
    return res


def get_str(a, b):
    a = str(a)
    b = str(b)
    return a + '(' + b + ')'


def get_anova():
    control = {}
    patient = {}
    for band in const.bands_name:
        control[band] = average_list(band, 'control')
        patient[band] = average_list(band, 'patient')

    title = ['Frontal', 'Left_Frontal', 'Right_Frontal']
    df = pd.DataFrame(columns= title,index=const.bands_name)
    for band in const.bands_name:
        temp = []
        f, p = f_oneway(control[band]['all_kfd'], patient[band]['all_kfd'])
        temp.append(get_str(p,f))
        f, p = f_oneway(control[band]['left_kfd'], patient[band]['left_kfd'])
        temp.append(get_str(p,f))
        f, p = f_oneway(control[band]['right_kfd'], patient[band]['right_kfd'])
        temp.append(get_str(p,f))
        df.loc[band] = temp
    df.to_csv('csv/kfd_anova.csv')

    df = pd.DataFrame(columns= title,index=const.bands_name)
    for band in const.bands_name:
        temp = []
        f, p = f_oneway(control[band]['all_hfd'], patient[band]['all_hfd'])
        temp.append(get_str(p,f))
        f, p = f_oneway(control[band]['left_hfd'], patient[band]['left_hfd'])
        temp.append(get_str(p,f))
        f, p = f_oneway(control[band]['right_hfd'], patient[band]['right_hfd'])
        temp.append(get_str(p,f))
        df.loc[band] = temp
    df.to_csv('csv/hfd_anova.csv')



get_csv()
get_anova()

