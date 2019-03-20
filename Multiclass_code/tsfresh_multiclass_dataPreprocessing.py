import pandas as pd
import mne
import numpy as np
import os

base_path = '../Multiclass/'


# 从patient_info文件中整理出我们想要的信息
def handle_xlsx(name):
    data_xls = pd.read_excel(base_path + name, index_col=0)

    df = pd.DataFrame(columns=['id', 'types', 'file'])

    for index, row in data_xls.iterrows():
        temp = [str(row['Patient_No']).split('_')[0]]
        if str(row['Score']) == 'nan':
            print(index)
            continue
        if int(row['Score']) <= 7:
            temp.append('0')
        elif 8 <= int(row['Score']) <= 13:
            temp.append('1')
        elif 14 <= int(row['Score']) <= 19:
            temp.append('2')
        elif 20 <= int(row['Score']) <= 25:
            temp.append('3')
        elif int(row['Score']) >= 26:
            temp.append('4')
        temp.append(row['File_Name'])
        df.loc[len(df)] = temp

    df.to_csv(base_path + 'info.csv', index=False)


# 获取文件中的关键信息
def get_info():
    return pd.read_csv(base_path + 'info.csv')


# 获取列名和坏的通道
def raw_data_info(filePath='/home/rbai/eegData'):
    raw = mne.io.read_raw_brainvision(filePath + '/health_control/eyeclose/jkdz_cc_20180430_close.vhdr',
                                      preload=True)
    # channel_names = raw.info['ch_names']
    print()
    channel_names = []
    for i in raw.info['ch_names']:
        if i != 'Oz':
            if i != 'ECG':
                channel_names.append(i)

    bad_channels = ['Oz', 'ECG']
    return channel_names, bad_channels


# 根据文件名读取raw
def get_raw(fname, badchannels, path='/home/rbai/eegData/mdd_patient/eyeclose/'):
    raw = mne.io.read_raw_brainvision(path + '/' + fname, preload=True)
    raw = raw.filter(None, 40)
    print('filter success')

    raw.resample(160, npad='auto')
    print('resample success')

    raw.info['bads'] = badchannels
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

    raw = raw.get_data(picks)
    raw = np.array(raw).T

    return raw


# 对每一个样本调整至我们想要的格式
def handle_entity(fname, badchannels, type, count):
    res = []
    time = 0.0
    raws = get_raw(fname, badchannels)
    for raw in raws:
        x = list(raw)
        x.insert(0, time)
        x.insert(0, count)
        time += 0.001
        x.append(type)
        res.append(x)
    return res


# 入口主函数
def dataPreprocessing():
    # handle_xlsx('patient_info.xlsx')
    pd_info = get_info()
    columns, bad_channels = raw_data_info()
    columns.insert(0, 'time')
    columns.insert(0, 'id')
    columns = columns[:-1]
    columns.append('y')
    count = 0
    print(pd_info)
    error = []
    for index, row in pd_info.iterrows():
        try:
            temp = handle_entity(str(row['file']), bad_channels, str(row['types']), count)
            print(count)
            res = pd.DataFrame(temp, columns=columns)
            res.to_csv(base_path + 'data/multiclass_180s_data_' + str(row['id']) + '_' + str(count) + '.csv',
                       index=False)
            count += 1

        except Exception:
            error.append(str(row['file']))

    print(error)


# 将每个样本的csv合成一个大的csv
def get_csv():
    path = '/home/rbai/Multiclass/data'  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    base = pd.read_csv(path + '/' + files[0])
    y=pd.DataFrame(columns=['y'])
    y.loc[len((y))]=list(base['y'])[0]
    count = 0
    for file in files:
        if count == 0:
            count += 1
            continue
        temp = pd.read_csv(path + '/' + file)
        base = base.append(temp)
        y.loc[len((y))] = list(temp['y'])[0]
        count += 1
        print(count)

    y.to_csv(base_path + 'multiclass_180s_y.csv',index=False)
    print('y success')
    base.to_csv(base_path + 'multiclass_180s_data.csv',index=False)


# dataPreprocessing()
get_csv()