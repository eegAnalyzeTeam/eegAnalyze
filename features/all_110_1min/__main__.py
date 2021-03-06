from features.all_110_1min import eeg_tsfresh_all_1min
from features.all_110_1min import thread_cal_features
from features.all_110_1min import eeg_tsfresh_selectFeatures
from features.all_110_1min import handle_multi_csv
from features.all_110_1min import eeg_k_cv
from features.all_110_1min import eeg_classify_model
from features.all_110_1min import test_k_cv_3
from features.all_110_1min import svm_curve

from features.all_110_1min import eeg_analyze_channel
from features.all_110_1min import test_k_cv_3_p

eeg_tsfresh_all_1min.start()  # 处理原始数据
thread_cal_features.start()  # 计算特征
eeg_tsfresh_selectFeatures.start()  # 选取特征
handle_multi_csv.start()  # 将多个个体合并
eeg_k_cv.start()  # k折交叉验证
eeg_classify_model.start()  # 30次随机序列，划分训练集测试集
test_k_cv_3.start()  # k折交叉验证，精确率、召回率、准确率
svm_curve.start()  # 学习曲线

eeg_analyze_channel.start()  # 对选取的特征p值排序
test_k_cv_3_p.start()  # p值排序后选取少量的特征分类
