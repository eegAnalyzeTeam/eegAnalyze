import numpy as np
import pandas as pd
import csv
import traceback


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


def get_svm_y():
    content = []

    for i in range(0, 1913):
        content.append(0)

    for i in range(1913, 8082):
        content.append(1)

    content = np.array(content).T
    df = pd.DataFrame(content)
    df.to_csv('svm_y.csv', header=0)


def start():
    try:
        get_svm_y()
        read_3second_extracedFeatures_numpy()
    except Exception as e:
        print(str(e))
        traceback.print_exc()
