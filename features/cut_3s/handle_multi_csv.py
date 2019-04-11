import numpy as np
import pandas as pd
import csv
import traceback


def read_alpha2():
    base = pd.read_csv('features_change/alpha2/control_data_' + str(0) + '.csv')
    for i in range(1, 30):
        temp = pd.read_csv('features_change/alpha2/control_data_' + str(i) + '.csv')
        base = base.append(temp)
    for i in range(30, 111):
        temp = pd.read_csv('features_change/alpha2/patient_data_' + str(i) + '.csv')
        base = base.append(temp)

    base.to_csv('features_change/alpha2/tsfresh_data.csv')


def read_alpha1():
    base = pd.read_csv('features_change/alpha1/control_data_' + str(0) + '.csv')
    for i in range(1, 30):
        temp = pd.read_csv('features_change/alpha1/control_data_' + str(i) + '.csv')
        base = base.append(temp)
    for i in range(30, 111):
        temp = pd.read_csv('features_change/alpha1/patient_data_' + str(i) + '.csv')
        base = base.append(temp)

    base.to_csv('features_change/alpha1/tsfresh_data.csv')


def read_all():
    base = pd.read_csv('features_change/all/control_data_' + str(0) + '.csv')
    for i in range(1, 30):
        temp = pd.read_csv('features_change/all/control_data_' + str(i) + '.csv')
        base = base.append(temp)
    for i in range(30, 111):
        temp = pd.read_csv('features_change/all/patient_data_' + str(i) + '.csv')
        base = base.append(temp)

    base.to_csv('features_change/all/tsfresh_data.csv')


def read_alpha1_extractedFeatures():
    lack_alpha1 = [8, 15, 28]
    base = pd.read_csv('features_change/alpha1/tsfresh_extractedFeatures' + str(0) + '.csv')
    for i in range(1, 111):
        if i in lack_alpha1:
            continue
        temp = pd.read_csv('features_change/alpha1/tsfresh_extractedFeatures' + str(i) + '.csv')
        base = base.append(temp)
        print('alpha1 ' + str(i))

    base.to_csv('features_change/alpha1/extracted_features.csv')


def read_alpha2_extractedFeatures():
    lack_alpha2 = [11, 21, 28, 53, 93, 95]
    base = pd.read_csv('features_change/alpha2/tsfresh_extractedFeatures' + str(0) + '.csv')
    for i in range(1, 111):
        if i in lack_alpha2:
            continue
        temp = pd.read_csv('features_change/alpha2/tsfresh_extractedFeatures' + str(i) + '.csv')
        base = base.append(temp)
        print('alpha2 ' + str(i))

    base.to_csv('features_change/alpha2/extracted_features.csv')


def read_allcut_extracedFeatures():
    lack_cutall = [0, 15, 77]
    base = pd.read_csv('tsfresh_extractedFeatures' + str(1) + '.csv')
    for i in range(2, 111):
        if i in lack_cutall:
            continue
        temp = pd.read_csv('tsfresh_extractedFeatures' + str(i) + '.csv')
        base = base.append(temp)
        print('allcut ' + str(i))

    base.to_csv('tsfresh_extractedFeatures.csv')


def read_1second_extracedFeatures():
    base = pd.read_csv('tsfresh_extractedFeatures' + str(0) + '.csv')
    for i in range(1, 24363):
        temp = pd.read_csv('tsfresh_extractedFeatures' + str(i) + '.csv')
        base = base.append(temp)
        print('all ' + str(i))

    base.to_csv('tsfresh_extractedFeatures.csv')


def read_1second_extracedFeatures_numpy():
    res = []
    csv_file = open('tsfresh_extractedFeatures' + str(0) + '.csv')  # 打开csv文件
    csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
    csv_reader_lines = list(csv_reader_lines)

    res.append(csv_reader_lines[0])
    res.append(csv_reader_lines[1])
    for i in range(1, 24363):
        csv_file = open('tsfresh_extractedFeatures' + str(i) + '.csv')  # 打开csv文件
        csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
        csv_reader_lines = list(csv_reader_lines)

        res.append(csv_reader_lines[1])
        print('all ' + str(i))

    fileread = open('tsfresh_extractedFeatures.csv', 'w', newline='')
    writer = csv.writer(fileread)
    writer.writerows(res)
    fileread.close()


def read_3second_extracedFeatures_numpy():
    # base = np.loadtxt('tsfresh_extractedFeatures' + str(0) + '.csv',delimiter=',')
    # res = np.array(base)
    res = []
    csv_file = open('tsfresh_extractedFeatures' + str(0) + '.csv')  # 打开csv文件
    csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
    csv_reader_lines = list(csv_reader_lines)

    res.append(csv_reader_lines[0])
    res.append(csv_reader_lines[1])
    for i in range(1, 8082):
        csv_file = open('tsfresh_extractedFeatures' + str(i) + '.csv')  # 打开csv文件
        csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
        csv_reader_lines = list(csv_reader_lines)

        res.append(csv_reader_lines[1])
        print('all ' + str(i))

    fileread = open('tsfresh_extractedFeatures.csv', 'w', newline='')
    writer = csv.writer(fileread)
    writer.writerows(res)
    fileread.close()


def read_3second_extracedFeatures():
    base = pd.read_csv('tsfresh_extractedFeatures' + str(0) + '.csv')
    for i in range(1, 8082):
        temp = pd.read_csv('tsfresh_extractedFeatures' + str(i) + '.csv')
        base = base.append(temp)
        print('all ' + str(i))

    base.to_csv('tsfresh_extractedFeatures.csv')


def get_svm_y():
    content = []

    for i in range(0, 1913):
        content.append(0)

    for i in range(1913, 8082):
        content.append(1)

    content = np.array(content).T
    df = pd.DataFrame(content)
    df.to_csv('svm_y.csv', header=0)


try:
    get_svm_y()
    read_3second_extracedFeatures_numpy()
except Exception as e:
    print(str(e))
    traceback.print_exc()
