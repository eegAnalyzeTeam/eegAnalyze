from math import sqrt

import mne
import numpy


# 定义检验标准
class Compare:
    def __init__(self, amodel, data):
        self.model = amodel  # 加载模型数据
        self.data = data  # 加载待分析数据
        self.count = self.fitcount(self.model, self.data)  # 计算符合模型的区间个数

    # 皮尔逊相关系数
    @staticmethod
    def pearson(list1, list2, num):
        sum1 = sum(list1)
        sum2 = sum(list2)
        sqsum1 = sum(pow(num, 2) for num in list1)
        sqsum2 = sum(pow(num, 2) for num in list2)
        mulsum = sum(list1[k] * list2[k] for k in range(num))
        son = mulsum - sum1 * sum2 / num
        mot = sqrt((sqsum1 - pow(sum1, 2) / num) * (sqsum2 - pow(sum2, 2) / num))
        if mot == 0:
            r = 0
        else:
            r = son / mot
        return r

    def fitcount(self, train, data):
        count = 0
        j = 0
        while (len(data) - j) >= len(train):
            d = data[j:j + len(train)]
            if (max(d) - min(d)) / 2 > 2:  # 振幅大于2才检验，提高效率
                train_tmp = train * (max(d) - min(d)) / 2
                # 振动既可能先下后上，也可能先上后下
                r = max(self.pearson(train_tmp, d, len(train_tmp)), self.pearson(0 - train_tmp, d, len(train_tmp)))
                if r > 0.8:  # 相关系数阈值设定
                    count += 1
                    print('r', r)
                    j += len(train)
                    continue
            j += int(len(train) / 2)  # 如果通过检验就跳过全部，否则只跳过一半，减小误差
        return count


raw = mne.io.read_raw_brainvision('Sample_files/jkdz_wlk_20180728_open.vhdr', preload=True)
raw.set_montage(mne.channels.read_montage("standard_1020"))
raw_tmp = raw.copy()
raw_tmp.filter(1, None, fir_design="firwin")
ica = mne.preprocessing.ICA(method="fastica")
ica.fit(raw_tmp, stop=raw.times[-1])
# 以上为之前任务

# 加载眼电模型
model = numpy.loadtxt('eeg_eyemvmt_model.csv')

# 提取原始数据信息
sample = ica.get_sources(inst=raw_tmp, start=0, stop=raw_tmp.times[-1]).get_data().copy()

content = []
for i in range(64):
    content.append(Compare(model, sample[i]).count)
    print(i)
    if content[-1] > 5:  # 通过检验数大于5即可接受
        break
bad_eog = len(content) - 1
print('bad_eog is {}'.format(bad_eog))

# 以下为之前工作
ica.exclude = [bad_eog]
ica.apply(raw)
raw.plot(n_channels=64, start=0, duration=raw.times[-1], scalings=dict(eeg=2e-4))
