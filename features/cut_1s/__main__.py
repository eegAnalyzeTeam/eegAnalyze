from features.cut_1s import all_dataGen
from features.cut_1s import all_calculate_features
from features.cut_1s import all_select_features
from features.cut_1s import handle_multi_csv
from features.cut_1s import test_k_cv_3
from features.cut_1s import eeg_classify_model
from features.cut_1s import all_classify_model_c_k
from features.cut_1s import all_learn_curve
from features.cut_1s import eeg_analyze_features

from features.cut_1s import eeg_analyze_anova_p
from features.cut_1s import test_k_cv_3_p

all_dataGen.start()  # 处理原始数据
all_calculate_features.start()  # 计算特征
all_select_features.start()  # 选取特征
handle_multi_csv.start()  # 将单个个体的特征合成一个总表
test_k_cv_3.start()  # 交叉验证
eeg_classify_model.start()  # 对某些情况进一步验证（30个随机序列划分测试集训练集训练）
all_classify_model_c_k.start()  # 利用70%数据交叉验证，另30%数据重新验证
all_learn_curve.start()  # 学习曲线
eeg_analyze_features.start()  # 对选取的通道进一步分析

eeg_analyze_anova_p.start()  # 将表现好的情况的特征按p值排序
test_k_cv_3_p.start()  # 根据p值排序结果，用更少的特征排序
