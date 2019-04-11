 # coherence_code文件夹
 ## python -m coherence_code 调用
这是计算coherence的主要python文件。
- [ ] eeg_classify_model.py
  - [ ] 主要用来分类
- [ ] eeg_coherence.py
  - [ ] 主要用来计算coherence
  - [ ] 计算取平均值，最后存储为一个csv
- [ ] eeg_coherence_anova.py
  - [ ] 需要用到 eeg_coherence.py 的结果
  - [ ] 计算 anova
 - [ ] eeg_coherence_anova_plot.py
    - [ ] 用来画anova的图像,包括正常图片的灰色图
 - [ ] eeg_coherence_each.py
    - [ ] 计算每一个人的coherence
 - [ ] eeg_coherence_plot.py
    - [ ] 画出 eeg_coherence.py 的结果
 - [ ] eeg_coherence_plot_difference.py
    - [ ] 画出差异较大的coherence
 - [ ] pick_eeg_coherence.py
    - [ ] 挑选差异比较大的通道对合成一张表
    - [ ] 主要用来分类使用

### 代码如果不能正常运行请检查以下两个方面

- [x] 代码路径中的文件或文件夹是否存在或者路径是否正确
- [x] import的py文件是否在正确路径
