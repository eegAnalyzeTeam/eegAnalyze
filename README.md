# eegAnalyze - 寻找与抑郁症相关的脑电特征

本仓库实现抑郁症脑电数据的预处理、特征计算、可视化、有效特征发现等功能。

## 数据来源

安定医院抑郁症实验的静息态脑电数据(\\public2\\eegData)。

### 数据信息

| 参数 | 值|
|---|---|
| 仪器名称：| BrainVision Recoder |
| 通道数 :| 64|
| 采样率： | 5000 Hz|
| 分辨率 | 0.5uV |
| 高通滤波 | 0.1 Hz|
| 低通滤波 | 250 Hz|
| 工频陷波 | 无 |
| 采样阻抗 | 10 kΩ|

### 样本信息

| | 睁眼 | 闭眼 |
|---|---|---|
| MDD 组 | 84 | 87 |
| Control 组 | 27 | 30 |

## 数据预处理

主要依赖于[MNE软件包](https://mne-tools.github.io)进行。使用mne内置的mne.raw等数据结构。

- [x] 导入数据, `mne.io.read_raw`，以下过程中均对mne.raw对象操作
- [x] 标记坏通道, `raw.info['bads']`
- [x] 坏通道插值, `interpolate_bads`
- [ ] 标记坏段, `mne.Annotations`
- [x] 去除工频噪声, `raw.notch_filter`
- [x] 去除漂移, `raw.filter`
- [x] 去眼电/心电
  ``` python
  ica_model = get_artifact_model(mne_raw, ica_method='fastica', type = ['ECG', 'EOG'], plot = True, update_model = False)

  mne_raw = apply_arifact_model(mne_raw, ica_model)
  ```
- [x]  降采样, `raw.resample(256, npad="auto")`
- [x]  可视化, `raw.plot`, `raw.plot_sensors`, `raw.plot_psd`, ...
- [x] 保存数据, `raw.save`, `ica.save`,...

## 特征计算
以下过程均使用 pandas Dataframe 数据结构，可通过mne.raw得到。

### 时域

主要依赖于[tsfresh软件包](https://tsfresh.readthedocs.io)进行。

所计算的特征见[官网](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)
。

### 频域
主要使用 mne.time_frequency 计算。
- [x] 频谱
  - [x] 使用fft
  - [ ] 使用小波变换 hwt
  - [ ] 使用AR模型
- [ ] bicoherence, bispectrum, 使用 [Higher Order Spectrum Estimation toolkit](https://github.com/synergetics/spectrum), **哪些频率组合是有效特征需要进一步考察**

### 时频域

对时间序列分解后再使用以上的时域、频域方法。分解使用 [pywavelets](https://pywavelets.readthedocs.io/en/latest/) 或 [PyEMD](https://github.com/laszukdawid/PyEMD)。

- [ ] EEMD
- [ ] Hilbert-Huang spectrum
  > 暂无python实现，需考查是否必要

### 非线性/信息论

使用[tsfresh软件包](https://tsfresh.readthedocs.io)、[nolds](https://pypi.org/project/nolds/)、[pyEntropy](https://github.com/nikdon/pyEntropy)等进行。

- [x] approximate entropy (tsfresh & nolds)
- [X] sample entropy ( tsfresh & nolds & pyEntropy)
- [x] correlation dimension (nolds)
- [x] Lyapunov exponent (nolds)
- [x] Hurst exponent (nolds)
- [x] detrende fluctuation analysis (nolds)
- [X] permutation entropy (pyEntropy)
- [x] multiscale entropy (pyEntropy)
- [X] multiscale permutation entropy (pyEntropy)

> approximate entropy 与 sample entropy 类似，推荐使用后者；如果sample entropy或permutation entropy效果好，考虑进一步使用multiscale 版本提升效果，否则跳过multiscale版本。

### 双通道、网络

线性方法：
- [ ] cross correlation 
- [ ] cross covariance
- [ ] cross coherence, `mne.connectivity.spectral_connectivity`
    - [x] coherency
    - [ ] crossspectra, 使用 [Higher Order Spectrum Estimation toolkit](https://github.com/synergetics/spectrum)
    - [ ] wavelet transform based
- [ ] transfer function
- [ ] Granger causality
- [x] phase locking value, `mne.connectivity.spectral_connectivity`

非线性方法：
- [ ] nonlinear interdependence 
- [ ] cross recurrence
- [ ] nonlinear correlation

信息熵方法：
[pyitlib](https://pafoster.github.io/pyitlib/)
- [X] mutual information (`sklearn.metrics.mutual_info_score`, pyitlib)
- [ ] correlation entropy
- [ ] transfer entropy
- [x] cross entropy(pyitlib)
- [ ] 

复杂网络分析：
使用[graph-tool](https://graph-tool.skewed.de/static/doc/index.html)

- [ ] S-estimator
- [x] centrality
- [x] clustering coefficient

## 特征选择

- [ ] TPOT?
- [ ] feature_selector?
- [ ] feature_tools
- [ ] flap确定特征权重?

## 建模

- [ ] imbalanced-learn解决样本不匹配的问题?
- [ ] 模型特征重要性解释？ `XGBoost`, `SHAP`, `[ELI5](https://github.com/TeamHG-Memex/eli5)`, `skater`
    > Ref: https://towardsdatascience.com/human-interpretable-machine-learning-part-1-the-need-and-importance-of-model-interpretation-2ed758f5f476


## 加速

- [ ]  XGBoost?

## 可视化平台

基于 Dash 实现的数据分析控制台。

- [ ] 原始数据可视化
- [ ] 统计可视化
- [ ] 频谱、时频谱
- [ ] 空间可视化
  - [ ] 地形图
- [ ] 机器学习
  - [ ] 流程步骤选择
  - [ ] feature 参数选择
  - [ ] 学习方法选择
  - [ ] 结果性能可视化
- [ ] 图形导出/保存
- [ ] 结果导出/保存

## 应用
RESTful API风格的应用服务。

### 抑郁预测

## 代码结构

### psd_code
 
这是计算psd的主要python文件。
- [ ] eeg_psd_anova.py
  - [ ] 主要用来计算psd的anova
- [ ] eeg_psd_channel.py
  - [ ] 主要用来挑选差异比较大的通道
- [ ] eeg_psd_csv.py
    - [ ] 主要用来计算alpha1、alpha2、beta、gamma、delta等psd并存储csv文件
 - [ ] eeg_psd_plot.py
    - [ ] 用来画psd的图像
 - [ ] eeg_svm_classify.py
    - [ ] svm的分类器，基本弃用。
    - [ ] 分类可以使用classify目录下的eeg_classify_model.py
 - [ ] eegAnalyze.py
    - [ ] 用来读取eeg信息的文件
 
 
 ### coherence_code
 
这是计算coherence的主要python文件。
- [ ] eeg_coherence.py
  - [ ] 主要用来计算coherence
  - [ ] 计算取平均值，最后存储为一个csv
- [ ] eeg_coherence_anova.py
  - [ ] 需要用到 eeg_coherence.py 的结果
  - [ ] 计算 anova
- [ ] eeg_psd_csv.py
    - [ ] 主要用来计算alpha1、alpha2、beta、gamma、delta等psd并存储csv文件
 - [ ] eeg_coherence_anova_plot.py
    - [ ] 用来画anova的图像,包括正常图片的灰色图
 - [ ] eeg_coherence_each.py
    - [ ] 计算每一个人的coherence
 - [ ] eeg_coherence_plot.py
    - [ ] 画出 eeg_coherence.py 的结果
 - [ ] eeg_coherence_plot_difference.py
    - [ ] 画出差异较大的coherence
 - [ ] pick_eeg_coherence.py
    - [ ] 将eeg_coherence_each.py的结果挑选差异比较大的通道对合成一张表
    - [ ] 主要用来分类使用


