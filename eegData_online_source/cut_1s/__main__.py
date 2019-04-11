from eegData_online_source.cut_1s import eeg_calcFeatures
from eegData_online_source.cut_1s import all_select_features
from eegData_online_source.cut_1s import handle_multi_csv
from eegData_online_source.cut_1s import all_classify_model_c_k
from eegData_online_source.cut_1s import test_k_cv_3
from eegData_online_source.cut_1s import all_learn_curve
from eegData_online_source.cut_1s import eeg_analyze_features

eeg_calcFeatures.start()  # 计算特征
all_select_features.start()  # 选取特征
handle_multi_csv.start()  # 将多个个体合成一个整表
all_classify_model_c_k.start()  # k折交叉验证
test_k_cv_3.start()  # 利用70%数据交叉验证，另30%数据重新验证
all_learn_curve.start()  # 学习曲线
eeg_analyze_features.start()  # 对好的特征进行分析
