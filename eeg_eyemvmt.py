#!/usr/bin/env python
# coding: utf-8


import mne

raw = mne.io.read_raw_brainvision('Sample_files/jkdz_mwy_20180429_open.vhdr', preload=True)
raw.set_montage(mne.channels.read_montage("standard_1020"))

raw_tmp1 = raw.copy()
print(raw_tmp1)

raw_tmp1.filter(1, None, fir_design="firwin")

ica = mne.preprocessing.ICA(method="fastica")

ica.fit(raw_tmp1, stop=raw_tmp1.times[-1])

ica.plot_components()
# 这里打印出64个大脑图像以便确认后续判断眼电伪迹的正确性

# 以上与第一周工作一样


# 作独立备份（不共用内存），为后续步骤作准备
ica_tmp1 = ica.copy()
ica_tmp2 = ica.copy()
raw_tmp2 = raw_tmp1.copy()

from mne.preprocessing import create_eog_epochs

# 以下分别选取Fp1，Fp2作为判断是否为眼电伪迹的标准，最终将最符合的合并起来作为最终眼电伪迹处理对象
eog_average = create_eog_epochs(raw_tmp1, ch_name='Fp1').average()
eog_epochs = create_eog_epochs(raw_tmp1, ch_name='Fp1')
eog_inds, scores = ica_tmp1.find_bads_eog(eog_epochs, ch_name='Fp1')

ica_tmp1.plot_sources(eog_average, exclude=eog_inds)
# 由图中可确定出哪一个是眼电伪迹（即陡峰），这里点击陡峰图线为ICA011
# 与大脑图像也相符合


print(ica_tmp1.labels_)  # 注意'eog'中第0个就是刚才的最高峰了，判断为函数已经将重要程度进行过排序

eog_average = create_eog_epochs(raw_tmp2, ch_name='Fp2').average()
eog_epochs = create_eog_epochs(raw_tmp2, ch_name='Fp2')
eog_inds, scores = ica_tmp2.find_bads_eog(eog_epochs, ch_name='Fp2')

ica_tmp2.plot_sources(eog_average, exclude=eog_inds)

print(ica_tmp2.labels_)

# 合并两次判断中'eog‘的第0个值作为最后的结果
blink = list(set([ica_tmp1.labels_['eog'][0], ica_tmp2.labels_['eog'][0]]))
print(blink)

# 以下同第一周
ica.exclude = blink

raw_corrected = raw.copy()
print(raw_corrected)

ica.apply(raw_corrected)

raw.plot(n_channels=64, start=0, duration=raw.times[-1], scalings=dict(eeg=250e-6))

raw_corrected.plot(n_channels=64, start=0, duration=raw_corrected.times[-1], scalings=dict(eeg=250e-6))

# 去除效果不错
# 后续可结合eeg_bad_channel.py除去bad channel

# 剩余的问题：
# 1. 仅选取Fp1，Fp2作为衡量标准可能有所偏差，比如其实也有其他的看起来像眼电伪迹的部分并没有被筛选出来
# 2. 还是过于依赖已有的算法，有待进一步加深理解
# 3. 本试图通过低通滤波来实现，但一方面由于预备知识的欠缺，另一方面缺少相关资料和程序，故最终未能实现，在以后可以补充
