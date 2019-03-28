# 这是对110个人的alpha1频段的信号，利用tsfresh库计算特征并进行分类的主要python文件
- [ ] eeg_analyze_channel.py
    - [ ] 将特征按p值排序（test）
- [ ] eeg_classify_model.py
    - [ ] 用来对选出的特征进行分类
- [ ] eeg_features_alpha1.py
    - [ ] 用来对信号选择alpha1频段和降采样
    - [ ] 并分别保存csv文件
- [ ] eeg_k_cv.py
    - [ ] 对选出的特征做交叉验证的脚本
 - [ ] eeg_tsfresh_selectFeatures.py
    - [ ] 使用tsfresh库选择特征
    - [ ] 使用sklearn库选择特征
 - [ ] handle_multi_csv.py
    - [ ] 把计算好特征的csv合并
    - [ ] 只用到read_alpha1()函数
 - [ ] svm_curve.py
    - [ ] 用来画学习曲线
 - [ ] test_k_cv_3.py
    - [ ] 十折交叉验证后计算精确度，准确度以及召回率
 - [ ] test_k_cv_3_p.py
    - [ ] 对p值较小的部分特征交叉验证（test）
 - [ ] thread_cal_features.py
    - [ ] 使用tsfresh计算特征
 
    
#   脚本执行顺序如下（仅参考）：
- [ ] eeg_features_alpha1.py
- [ ] thread_cal_features.py
- [ ] handle_multi_csv.py
- [ ] eeg_tsfresh_selectFeatures.py
- [ ] eeg_classify_model.py
- [ ] eeg_k_cv.py
- [ ] test_k_cv_3.py
- [ ] svm_curve.py
