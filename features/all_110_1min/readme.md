# 这是对110个人的信号并截取1分钟信号，利用tsfresh库计算特征并进行分类的主要python文件
- [ ] eeg_classify_model.py
    - [ ] 用来对选出的特征进行分类
- [ ] eeg_k_cv.py
    - [ ] 对选出的特征做交叉验证的脚本
- [ ] eeg_tsfresh_1min.py
    - [ ] 用来对信号降采样和截取1分钟时间
    - [ ] 并分别保存csv文件
 - [ ] eeg_tsfresh_calcFeatures.py
    - [ ] 使用tsfresh库选择特征
    - [ ] 使用sklearn库选择特征
 - [ ] handle_multi_csv.py
    - [ ] 把计算好特征的csv合并
    - [ ] 只用到read_alpha1()函数
 - [ ] svm_curve.py
    - [ ] 用来画学习曲线
 - [ ] thread_cal_features.py
    - [ ] 使用tsfresh计算特征
 
    
 #    脚本执行顺序如下：
    - [ ] eeg_tsfresh_1min.py
    - [ ] thread_cal_features.py
    - [ ] handle_multi_csv.py
    - [ ] eeg_tsfresh_calcFeatures.py
    - [ ] eeg_classify_model.py
    - [ ] test_k_cv.py
    - [ ] svm_curve.py
