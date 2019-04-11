# 这是对58个人的信号并截取1分钟信号，利用tsfresh库计算特征并进行多级分类的主要python文件

## 调用命令   python -m Multiclass_code

- [ ] tsfresh_multiclass_calcFeatures.py
    - [ ] 对每个人计算特征
    - [ ] 将所有特征合成一个文件
- [ ] tsfresh_multiclass_classification.py
    - [ ] k折交叉验证
- [ ] tsfresh_multiclass_dataPreprocessing.py
    - [ ] 从文件中读取数据
    - [ ] 将脑电信号切割、降采样等
 - [ ] tsfresh_multiclass_learningCurve.py
    - [ ] 画学习曲线
 - [ ] tsfresh_multiclass_normal_classify.py
    - [ ] 针对表现好的情况，多次取0.8训练集的随机样本训练
 - [ ] tsfresh_multiclass_selectFeatures.py
    - [ ] 使用tsfresh库选择特征
    - [ ] 使用sklearn库选择特征
 
    
#  脚本执行顺序如下（仅参考）：
- [ ] tsfresh_multiclass_dataPreprocessing.py
- [ ] tsfresh_multiclass_calcFeatures.py
- [ ] tsfresh_multiclass_selectFeatures.py
- [ ] tsfresh_multiclass_classification.py
- [ ] tsfresh_multiclass_normal_classify.py
- [ ] tsfresh_multiclass_learningCurve.py
