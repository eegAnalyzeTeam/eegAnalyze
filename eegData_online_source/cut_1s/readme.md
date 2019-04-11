# 这是对分割成1s的信号，利用tsfresh库计算特征并进行分类的主要python文件

## 调用命令： python -m eegData_online_source.cut_1s

- [ ] all_classify_model_c_k.py
    - [ ] 首先把数据(对表现好的情况)按照0.3、0.7的比例分割，在0.7的数据部分进行十折交叉验证
    - [ ] 选出交叉验证时表现比较好的模型
    - [ ] 在0.3的数据部分用选出的模型进行预测，并计算准确度
- [ ] all_learn_curve.py
    - [ ] 用来画学习曲线
- [ ] all_select_features.py
    - [ ] 使用sklearn库选择特征
    - [ ] 使用tsfresh库选择特征
- [ ] eeg_analyze_features.py
    - [ ] 提取选到的特征主要是get_colums_tree()函数
- [ ] eeg_calcFeatures.py
    - [ ] 用tsfresh计算特征
- [ ] eeg_getData.py
    - [ ] 读取文件eeg数据
- [ ] handle_multi_csv.py
    - [ ] 把计算好特征的csv合并
    - [ ] 只用到get_svm_y()和test_read_1s()两个函数
- [ ] test_k_cv_3.py
    - [ ] 对选出的特征做10折交叉验证
    
#  脚本执行顺序如下：
- [ ] eeg_getData.py
- [ ] eeg_calcFeatures.py
- [ ] handle_multi_csv.py
- [ ] all_select_features.py
- [ ] test_k_cv_3.py
- [ ] all_curve.py
- [ ] all_classify_model_c_k.py
- [ ] eeg_analyze_features.py
