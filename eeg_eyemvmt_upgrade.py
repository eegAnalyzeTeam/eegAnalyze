import mne
from math import sqrt
import numpy


# 定义检验标准
class compare():
    def __init__(self, model, data):
        self.model = model  # 加载模型数据
        self.data = data  # 加载待分析数据
        self.count = self.fitcount(self.model, self.data)  # 计算符合模型的区间个数

    # 皮尔逊相关系数
    def pearson(self, T1, T2, cnt):
        sum1 = sum(T1)
        sum2 = sum(T2)
        sqSum1 = sum(pow(num, 2) for num in T1)
        sqSum2 = sum(pow(num, 2) for num in T2)
        mulSum = sum(T1[i] * T2[i] for i in range(cnt))
        son = mulSum - sum1 * sum2 / cnt
        mot = sqrt((sqSum1 - pow(sum1, 2) / cnt) * (sqSum2 - pow(sum2, 2) / cnt))
        if mot == 0:
            r = 0
        else:
            r = son / mot
        return r

    def fitcount(self, train, data):
        count = 0
        i = 0
        while (len(data) - i) >= len(train):
            d = data[i:i + len(train)]
            if (max(d) - min(d)) / 2 > 2:  # 振幅大于2才检验，提高效率
                train_tmp = train * (max(d) - min(d)) / 2
                # 振动既可能先下后上，也可能先上后下
                r = max(self.pearson(train_tmp, d, len(train_tmp)), self.pearson(-train_tmp, d, len(train_tmp)))
                if r > 0.8:  # 相关系数阈值设定
                    count += 1
                    print('r', r)
                    i += len(train)
                    continue
            i += int(len(train) / 2)
            # 如果通过检验就跳过全部，否则只跳过一半，减小误差
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
    content.append(compare(model, sample[i]).count)
    print(i)
    if content[-1] > 5:  # 通过检验数大于5即可接受
        break
bad_eog = len(content) - 1
print('bad_eog is {}'.format(bad_eog))

# 以下为之前工作
ica.exclude = [bad_eog]
ica.apply(raw)
raw.plot(n_channels=64, start=0, duration=raw.times[-1], scalings=dict(eeg=2e-4))
