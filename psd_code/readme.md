# psd_code文件夹
## 请在主目录下python -m psd_code调用此模块
 
### 这是计算psd的主要python文件。
- [ ] check_file.py
  - [ ] 主要用来检查该文件是否可以读取
- [ ] eegAnalyze.py
  - [ ] 用来读取eeg信息的文件
- [ ] eeg_classify_model.py
  - [ ] 实现了几种sklearn的分类方式
- [ ] eeg_psd_anova.py
  - [ ] 主要用来计算psd的anova
- [ ] eeg_psd_channel.py
  - [ ] 主要用来挑选差异比较大的通道
- [ ] eeg_psd_csv.py
  - [ ] 主要用来计算alpha1、alpha2、beta、gamma、delta等psd并存储csv文件
- [ ] eeg_psd_plot.py
  - [ ] 用来画psd的图像
- [ ] eeg_svm_classify.py
  - [ ] svm的分类器
    
### 调用顺序一般为

- [ ] eegAnalyze.py
- [ ] eeg_psd_csv.py
- [ ] eeg_psd_plot.py
- [ ] eeg_psd_anova.py
- [ ] eeg_svm_classify.py（eeg_classify_model.py）

    
  ### 代码如果不能正常运行请检查以下两个方面

- [x] 代码路径中的文件或文件夹是否存在或者路径是否正确
- [x] import的py文件是否在正确路径
