import mne
from mne.preprocessing import create_eog_epochs

raw = mne.io.read_raw_brainvision('Sample_files/jkdz_mwy_20180429_open.vhdr', preload=True)
raw.set_montage(mne.channels.read_montage("standard_1020"))
raw_tmp1 = raw.copy()
raw_tmp1.filter(1, None, fir_design="firwin")

ica = mne.preprocessing.ICA(method="fastica")
ica.fit(raw_tmp1, stop=raw_tmp1.times[-1])
# 以上与第一周工作一样

# 作独立备份（不共用内存），为后续步骤作准备
ica_tmp1 = ica.copy()
ica_tmp2 = ica.copy()
raw_tmp2 = raw_tmp1.copy()

# 以下分别选取Fp1，Fp2作为判断是否为眼电伪迹的标准，最终将最符合的合并起来作为最终眼电伪迹处理对象
eog_epochs = create_eog_epochs(raw_tmp1, ch_name='Fp1')
ica_tmp1.find_bads_eog(eog_epochs, ch_name='Fp1')

eog_epochs = create_eog_epochs(raw_tmp2, ch_name='Fp2')
ica_tmp2.find_bads_eog(eog_epochs, ch_name='Fp2')

# 合并两次判断中'eog‘的第0个值作为最后的结果
blink = list({[ica_tmp1.labels_['eog'][0], ica_tmp2.labels_['eog'][0]]})
print('bad_eog is {}'.format(blink))

# 以下同第一周
ica.exclude = blink
ica.apply(raw)
raw.plot(n_channels=64, start=0, duration=raw.times[-1], scalings=dict(eeg=2e-4))
