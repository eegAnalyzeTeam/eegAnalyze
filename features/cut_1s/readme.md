# 这是对分割成1s的信号，利用tsfresh库计算特征并进行分类的主要python文件
## 调用命令  python -m features.cut_1s

- [ ] all_calculate_features.py
    - [ ] 用来计算全部的特征
- [ ] all_classify_model_c_k.py
    - [ ] 首先把数据(含88个特征)按照0.3、0.7的比例分割，在0.7的数据部分进行十折交叉验证
    - [ ] 选出交叉验证时表现比较好的模型
    - [ ] 在0.3的数据部分用选出的模型进行预测，并计算准确度
- [ ] all_dataGen.py
    - [ ] 把信号降采样后，按3s分割。分别保存csv
- [ ] all_curve.py
    - [ ] 用来画学习曲线
- [ ] all_select_features.py
    - [ ] 使用sklearn库选择特征
    - [ ] 使用tsfresh库选择特征
- [ ] eeg_analyze_anova_p.py
    - [ ] 对选到的t特征按p值排序并保存csv
- [ ] eeg_analyze_features.py
    - [ ] 分析选区的特征和通道（主要是从csv文件中提取出来）
- [ ] eeg_classify_model.py
    - [ ] 0.7训练集0.3测试集计算分类，可以看到病人和正常人分类
- [ ] handle_multi_csv.py
    - [ ] 把计算好特征的csv合并
    - [ ] 只用到get_svm_y()和read_3second_extracedFeatures_numpy()两个函数
- [ ] test_k_cv_3.py
    - [ ] 对选出的特征做10折交叉验证
- [ ] test_k_cv_3_p.py
    - [ ] 再选取p值较小的几列做分类
    
#  脚本执行顺序如下（仅参考，可以根据实际有调整或者单独运行某个脚本）：
- [ ] all_dataGen.py
- [ ] all_calculate_features.py
- [ ] handle_multi_csv.py
- [ ] all_select_features.py
- [ ] test_k_cv_3.py
- [ ] eeg_classify_model.py
- [ ] all_curve.py
- [ ] eeg_analyze_features.py
- [ ] all_classify_model_c_k.py
- [ ] eeg_analyze_anova_p.py
- [ ] test_k_cv_3_p.py
