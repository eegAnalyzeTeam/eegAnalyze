import numpy as np
import pandas as pd
import mne

def get_colums(file_name):
    temp = np.loadtxt(file_name, dtype=str,delimiter=',')
    temp=map(lambda x:x.strip('"'),list(temp[0]))
    temp=list(temp)
    # temp = map(lambda x: x.split('"')[0], temp)
    return list(temp)


def analyze_colums(list1,list2,all):
    temp_features=[]
    for x in list1:
        for y in list2:
            if x == y:
                temp_features.append(x)
    # temp_res=[]
    # for x in all:
    #     if x in temp_features:
    #         temp_res.append(x)
    return temp_features


def get_channel(features):
    temp_res=[]
    for x in features:
        temp=str(x).split('_')[0]
        temp_res.append(temp)
    return list(set(temp_res))


def analyze_channel(features1,features2,features3):
    channel1=get_channel(features1)
    channel2=get_channel(features2)
    channel3=get_channel(features3)

    print('alpha1:')
    print(channel1)
    print(len(channel1))
    print('alpha2:')
    print(channel2)
    print(len(channel2))
    print('all:')
    print(channel3)
    print(len(channel3))

    alpha2=[]
    for x in channel1:
        if x not in channel2:
            alpha2.append(x)
    print('alpha2:')
    print(alpha2)

    all=[]
    for x in channel1:
        if x not in channel3:
            all.append(x)
    print('all:')
    print(all)


def get_all_colums():
    feature_alpha1=get_colums('test_sklearn_SelectFromModel_alpha1.csv')[1:]
    feature_alpha2=get_colums('test_sklearn_SelectFromModel_alpha2.csv')[1:]
    feature_all=get_colums('features/test_sklearn_SelectFromModel.csv')[1:]

    print(len(feature_alpha1))
    print(feature_alpha1)
    print(len(feature_alpha2))
    print(feature_alpha2)
    print(len(feature_all))
    print(feature_all)

    temp=analyze_colums(feature_all,feature_alpha1)
    temp=analyze_colums(temp,feature_alpha2)
    print(temp)
    df=pd.DataFrame(temp)
    df.to_csv('analyze_features.csv',index=False,header=False)
    analyze_channel(feature_alpha1,feature_alpha2,feature_all)

    df = pd.DataFrame(feature_alpha1)
    df.to_csv('analyze_features_alpha1.csv', index=False, header=False)
    df = pd.DataFrame(feature_alpha2)
    df.to_csv('analyze_features_alpha2.csv', index=False, header=False)
    df = pd.DataFrame(feature_all)
    df.to_csv('analyze_features_all.csv', index=False, header=False)



def analyze_channel_2(features1,features2,feature_all):
    channel1=get_channel(features1)
    channel2=get_channel(features2)
    all=get_channel(feature_all)

    print('alpha1:')
    print(channel1)
    print(len(channel1))
    print('alpha2:')
    print(channel2)
    print(len(channel2))
    print('all')
    print(all)
    print(len(all))


    alpha1=[]
    for x in channel1:
        if x not in channel2:
            alpha1.append(x)
    print('alpha1:')
    print(alpha1)

    alpha2=[]
    for x in channel2:
        if x not in channel1:
            alpha2.append(x)
    print('alpha2:')
    print(alpha2)

    print('lack channel;')
    channel_names, bad_channels=raw_data_info('../eegData')
    for x in channel_names:
        if x in channel2:
            pass
        else:
            print(x)

def get_all_colums_2019():
    feature_alpha1 = get_colums('alpha1/select_features_VarianceThreshold.csv')[1:]
    feature_alpha2 = get_colums('alpha2/select_features_VarianceThreshold.csv')[1:]
    feature_all=get_colums('all_cut/select_features_VarianceThreshold.csv')[1:]

    print(len(feature_alpha1))
    print(feature_alpha1)
    print(len(feature_alpha2))
    print(feature_alpha2)
    print(len(feature_all))
    print(feature_all)

    temp = analyze_colums(feature_alpha1, feature_alpha2,feature_all)

    print(temp)
    df = pd.DataFrame(temp)
    df.to_csv('analyze_features.csv', index=False, header=False)

    analyze_channel_2(feature_alpha1, feature_alpha2,feature_all)

    df = pd.DataFrame(feature_alpha1)
    df.to_csv('analyze_features_alpha1.csv', index=False, header=False)
    df = pd.DataFrame(feature_alpha2)
    df.to_csv('analyze_features_alpha2.csv', index=False, header=False)
    df = pd.DataFrame(feature_all)
    df.to_csv('analyze_features_all.csv', index=False, header=False)

def raw_data_info(filePath):
    raw = mne.io.read_raw_brainvision(filePath+'/health_control/eyeclose/jkdz_cc_20180430_close.vhdr',
                                      preload=True)
    #channel_names = raw.info['ch_names']
    print()
    channel_names = []
    for i in raw.info['ch_names']:
        if i!='Oz':
            if i!='ECG':
                channel_names.append(i)

    bad_channels = ['Oz','ECG']
    return channel_names, bad_channels


def start():
    get_all_colums_2019()