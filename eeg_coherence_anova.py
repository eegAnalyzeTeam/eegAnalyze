import csv

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import eeg_psd_channel
import mne
# This program reads all csv files and does the ANOVA test

def coh_get_channel_names():
    raw = mne.io.read_raw_brainvision('/home/public2/eegData/health_control/eyeclose/jkdz_cc_20180430_close.vhdr',
                                      preload=True)
    channel_names = []
    for i in raw.info['ch_names']:
        if i != 'Oz':
            if i != 'ECG':
                channel_names.append(i)
    return channel_names[0:-1]

def get_psd_dfs(colum1,colum2):
    c_file_name = 'eeg_coh_anova/'+str(colum1)+'_'+str(colum2)+'_c'+'.csv'
    p_file_name = 'eeg_coh_anova/'+str(colum1)+'_'+str(colum2)+'_p'+'.csv'
    c_df = pd.read_csv(c_file_name)
    p_df = pd.read_csv(p_file_name)

    df = pd.merge(c_df, p_df, how='outer')
    return df

def get_coherence_anova():
    _columns = coh_get_channel_names()
    # file = open('coh_anova_res.txt', 'w')

    Histogram=[] # 获取画柱状图的一些信息
    Hist_fileread = open('coh_anova_histogram.csv', 'w', newline='')
    Hist_writer = csv.writer(Hist_fileread)
    Hist_writer.writerow(['i','j','column_name1','column_name2','p_value','f_value'])

    map_anova_pvalue=[]
    map_anova_fvalue=[]
    for i in range(62):
        data_t_p = []
        data_t_f=[]
        for j in range(62):
            if 61-i==j:
                data_t_p+=[1]
                data_t_f+=[0]
                continue
            if (61-i)>j:
                df=get_psd_dfs(_columns[j],_columns[61-i])
            else:
                df = get_psd_dfs(_columns[61-i], _columns[j])
            anova_res = anova_lm(ols('coherence ~ C(id)', df).fit(), typ=1)
            data_t_p+=[anova_res.loc['C(id)']['PR(>F)']]
            data_t_f += [anova_res.loc['C(id)']['F']]
            if (61-i)>j:
                continue
            if anova_res.loc['C(id)']['PR(>F)']<0.01:
                print(str(_columns[61-i])+'_'+str(_columns[j])+'_' + ' ANOVA Result')
                print(anova_res)
                Hist_writer.writerow([61-i,j,str(_columns[61-i]),str(_columns[j]),anova_res.loc['C(id)']['PR(>F)'],anova_res.loc['C(id)']['F']])
            # df=get_psd_dfs(_columns[61-i],_columns[j])
            # anova_res = anova_lm(ols('coherence ~ C(id)', df).fit(), typ=1)
            # data_t += [anova_res.loc['C(id)']['PR(>F)']]
            # print(str(_columns[61-i])+'_'+str(_columns[j])+'_' + ' ANOVA Result')
            # print(anova_res)
            # file.writelines(str(_columns[61-i])+'_'+str(_columns[j])+'_' + ' ANOVA Result' + "\r\n")
            # file.writelines(str(anova_res))
        map_anova_pvalue.append(data_t_p)
        map_anova_fvalue.append(data_t_f)
        Hist_fileread.close
    return map_anova_fvalue,map_anova_pvalue,_columns,Histogram



    # file.close()
    # draw_heatmap(map_anova,list(_columns),list(_columns)[::-1],'anova_p')

def coh_anova_save_csv():
    anova_fvalue,anova_pvalue,_columns,hist=get_coherence_anova()
    fileread = open('coh_anova_pvalue.csv', 'w', newline='')
    writer = csv.writer(fileread)
    writer.writerow(list(_columns))
    for pvalue in anova_pvalue:
        writer.writerow(list(pvalue))
    fileread.close

    fileread = open('coh_anova_fvalue.csv', 'w', newline='')
    writer = csv.writer(fileread)
    writer.writerow(list(_columns))
    for fvalue in anova_fvalue:
        writer.writerow(list(fvalue))
    fileread.close

# get_coherence_anova()

# coh_anova_save_csv()