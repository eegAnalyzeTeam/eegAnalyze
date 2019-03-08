import numpy as np
import pandas as pd
import csv
import traceback

# 将alpha1的每个全部特征文件合并成一个文件
# 这里没用到
def read_alpha1_extractedFeatures():
    lack_alpha1=[8,15,28]
    base = pd.read_csv('features_change/alpha1/tsfresh_extractedFeatures' + str(0) + '.csv')
    for i in range(1, 111):
        if i in lack_alpha1:
            continue
        temp = pd.read_csv('features_change/alpha1/tsfresh_extractedFeatures' + str(i) + '.csv')
        base = base.append(temp)
        print('alpha1 '+str(i))

    base.to_csv('features_change/alpha1/extracted_features.csv')

# 将alpha2的每个全部特征文件合并成一个文件
# 这里没用到
def read_alpha2_extractedFeatures():
    lack_alpha2=[11,21,28,53,93,95]
    base = pd.read_csv('features_change/alpha2/tsfresh_extractedFeatures' + str(0) + '.csv')
    for i in range(1, 111):
        if i in lack_alpha2:
            continue
        temp = pd.read_csv('features_change/alpha2/tsfresh_extractedFeatures' + str(i) + '.csv')
        base = base.append(temp)
        print('alpha2 '+str(i))

    base.to_csv('features_change/alpha2/extracted_features.csv')


# 将all的每个全部特征文件合并成一个文件
# 这里没用到
def read_allcut_extracedFeatures():
    lack_cutall=[0,15,77]
    base = pd.read_csv('tsfresh_extractedFeatures' + str(1) + '.csv')
    for i in range(2, 111):
        if i in lack_cutall:
            continue
        temp = pd.read_csv('tsfresh_extractedFeatures' + str(i) + '.csv')
        base = base.append(temp)
        print('allcut '+str(i))

    base.to_csv('tsfresh_extractedFeatures.csv')


# 将1s的数据各个文件合并成一个文件
def read_1second_extracedFeatures_numpy():
    res = []
    csv_file = open('tsfresh_extractedFeatures' + str(0) + '.csv')  # 打开csv文件
    csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
    csv_reader_lines = list(csv_reader_lines)

    res.append(csv_reader_lines[0])
    res.append(csv_reader_lines[1])
    for i in range(1, 20000):
        csv_file = open('tsfresh_extractedFeatures' + str(i) + '.csv')  # 打开csv文件
        csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
        csv_reader_lines = list(csv_reader_lines)

        res.append(csv_reader_lines[1])
        print('all ' + str(i))

    fileread = open('tsfresh_extractedFeatures.csv', 'w', newline='')
    writer = csv.writer(fileread)
    writer.writerows(res)
    fileread.close()


# 将3s的数据各个文件合并成一个文件
# 这里没用到
def read_3second_extracedFeatures_numpy():
    # base = np.loadtxt('tsfresh_extractedFeatures' + str(0) + '.csv',delimiter=',')
    # res = np.array(base)
    res=[]
    csv_file = open('tsfresh_extractedFeatures' + str(0) + '.csv')  # 打开csv文件
    csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
    csv_reader_lines=list(csv_reader_lines)

    res.append(csv_reader_lines[0])
    res.append(csv_reader_lines[1])
    for i in range(1, 8082):
        csv_file = open('tsfresh_extractedFeatures' + str(i) + '.csv')  # 打开csv文件
        csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
        csv_reader_lines = list(csv_reader_lines)

        res.append(csv_reader_lines[1])
        print('all '+str(i))

    fileread = open('tsfresh_extractedFeatures.csv', 'w', newline='')
    writer = csv.writer(fileread)
    writer.writerows(res)
    fileread.close()


# 根据顺序构造出每个人对应的y值
def get_svm_y():
    content=[]

    # for i in range(0,1913):
    #     content.append(0)
    #
    # for i in range(1913,8082):
    #     content.append(1)

    for i in range(0,5774):
        content.append(0)

    for i in range(5774,20000):
        content.append(1)

    content=np.array(content).T
    df=pd.DataFrame(content)
    df.to_csv('svm_y.csv',header=0)


try:
    get_svm_y()
    read_1second_extracedFeatures_numpy()
except Exception as e:
    print(str(e))
    traceback.print_exc()
