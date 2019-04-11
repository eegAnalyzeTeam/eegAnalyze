import numpy as np
import pandas as pd
import csv
import traceback


def test_read_3s():
    res = []
    csv_file = open('data/tsfresh_extractedFeatures' + str(0) + '.csv')  # 打开csv文件
    csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
    csv_reader_lines = list(csv_reader_lines)

    res.append(csv_reader_lines[0])
    res.append(csv_reader_lines[1])
    for i in range(1, 2881):
        csv_file = open('data/tsfresh_extractedFeatures' + str(i) + '.csv')  # 打开csv文件
        csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
        csv_reader_lines = list(csv_reader_lines)

        res.append(csv_reader_lines[1])
        print('all ' + str(i))

    fileread = open('tsfresh_extractedFeatures.csv', 'w', newline='')
    writer = csv.writer(fileread)
    writer.writerows(res)
    fileread.close()


def y_3s():
    content = []

    for i in range(0, 1420):
        content.append(0)

    for i in range(1420, 2881):
        content.append(1)

    content = np.array(content).T
    df = pd.DataFrame(content)
    df.to_csv('svm_y.csv', header=0)


def start():
    try:
        y_3s()
        test_read_3s()
    except Exception as e:
        print(str(e))
        traceback.print_exc()
