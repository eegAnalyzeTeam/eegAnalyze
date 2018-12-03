# -*- coding: utf-8 -*-
"""
@author: Wu Jiaqi
"""

import mne
import os
import csv
import re

def getdate(str):#读取字符串中日期
    dl = re.findall(r'\d{4}\d{2}\d{2}',str)
    return(dl)

def create(path,eye,group,name): #path:所在路径，eye：眼睛闭合情况，group：对照组实验组，name：生成csv文件名
    dirs=os.listdir(path)
    filename=[]
    for i in dirs:
        if os.path.splitext(i)[1]==".eeg":
            filename.append(os.path.splitext(i)[0])
    fileread=open(name+'.csv','w',newline='')
    writer=csv.writer(fileread)
    writer.writerow(['group(control/patient)','id','date','eyes(open/close)', 'num_channel','duration(s)','events'])
    for i in filename:
        raw=mne.io.read_raw_brainvision(path+'\\'+i+'.vhdr')
        a=str(raw)
        b=a[a.index(':'):a.index(')')+1]
        data=[group,i,getdate(a),eye,b[b.index(':')+1:b.index('x')],b[b.index('(')+1:b.index(')')],raw.info['events']]
        writer.writerow(data)  
path='D:\eeg\eegData\health_control\eyeopen'
eye='open'
group='control'
name='eeg_info_co'
create(path,eye,group,name)
#由于目录只有两层，就没有按照文件递归来写，总共生成4个csv文件可以合并来得到最后结果