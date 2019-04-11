import pandas as pd
import numpy as np

# 根据 coh_anova_histogram.csv 选择
x_index=[58,56,56,55,54]
y_index=[61,57,58,56,55]


def get_value(file,x,y):
    data=pd.read_csv(file)
    temp=[]
    for i in range(len(x)):
        temp.append(data.iloc[61-x[i],y[i]])
    return temp


def get_file():
    res=[]
    for i in range(1,31):
        temp=get_value('eeg_coherence_c_' + str(i) + '.csv',x_index,y_index)
        res.append(temp)
        print(str(i)+'c')

    for i in range(1,82):
        temp = get_value('eeg_coherence_p_' + str(i) + '.csv', x_index, y_index)
        res.append(temp)
        print(str(i) + 'p')

    df=pd.DataFrame(np.array(res),columns=['FT9-CPz','P08-P07','P07-FT9','P07-TP8','TP8-TP7'])
    df.to_csv('coh_train.csv')
