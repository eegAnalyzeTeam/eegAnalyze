import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import eeg_psd_channel

# This program reads all csv files and does the ANOVA test

def get_psd_dfs(subBand):
    c_file_name = 'psd_c_' + subBand + '.csv'
    p_file_name = 'psd_p_' + subBand + '.csv'
<<<<<<< HEAD
    c_df = pd.read_csv(c_file_name)
    p_df = pd.read_csv(p_file_name)
    del c_df['Unnamed: 0']
    del p_df['Unnamed: 0']
    # p_df = p_df.get(picks)
    # p_df=p_df[0:80]
    # p_df['average']=p_df.mean(1)
    # p_df['groupId']=1
    # c_df = c_df.get(picks)
    # c_df['average'] = c_df.mean(1)
    # c_df['groupId'] = 0
    df = pd.merge(c_df, p_df, how='outer')
    # print(df)
    return df

=======
    c_df=pd.read_csv(c_file_name)
    p_df=pd.read_csv(p_file_name)
    del c_df['Unnamed: 0']
    del p_df['Unnamed: 0']
    df=pd.merge(c_df, p_df, how='outer')
    return df
>>>>>>> 4bab13275698359399887e26df574008c4b1323b

def psd_anova():
    subBands = ['delta', 'theta', 'alpha1', 'alpha2', 'beta', 'gamma']
    # picks = eeg_psd_channel.pick_channel()
    # picks.append('groupId')
    for band in subBands:
        df = get_psd_dfs(band)
        anova_res = anova_lm(ols('average ~ C(groupId)', df).fit(), typ=1)
        print(band + ' ANOVA Result')
<<<<<<< HEAD
        print(anova_res)
=======
        print(anova_res)
>>>>>>> 4bab13275698359399887e26df574008c4b1323b
