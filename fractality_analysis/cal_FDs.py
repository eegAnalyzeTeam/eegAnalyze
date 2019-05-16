#
# 利用entropy计算HFDs和KFDs
#
#


import get_bands
import const
import numpy as np
import mne
from entropy import higuchi_fd
from entropy import katz_fd

import csv


def handle_raw(raw):
    res = []
    for x in const.brain:
        temp = []
        temp.append(x)
        raw_temp = raw.copy().pick_channels(temp)
        data = raw_temp.get_data()[0]
        hfd = higuchi_fd(data)
        kfd = katz_fd(data)
        res.append(hfd)
        res.append(kfd)
    return res


def get_raws():
    control_dic, patient_dic = get_bands.get_bands()
    for band in const.bands_name:
        print(band)
        res = []
        res.append('id')
        for x in const.brain:
            res.append(x + "_HFD")
            res.append(x + "_KFD")
        res.append('type')
        ress = []
        ress.append(res)
        count = 0
        for dic in control_dic:
            raw = dic[band]
            temp = handle_raw(raw)
            temp.insert(0, count)
            temp.append('0')
            ress.append(temp)
            count += 1

        fileread = open('csv/' + band + '_control.csv', 'w', newline='')
        writer = csv.writer(fileread)
        writer.writerows(ress)
        fileread.close()

        ress = []
        ress.append(res)
        count = 0
        for dic in patient_dic:
            raw = dic[band]
            temp = handle_raw(raw)
            temp.insert(0, count)
            temp.append('1')
            ress.append(temp)
            count += 1

        fileread = open('csv/' + band + '_patient.csv', 'w', newline='')
        writer = csv.writer(fileread)
        writer.writerows(ress)
        fileread.close()


get_raws()
