# BiLSTM_CNN-CRF-CWS
使用基于样本迁移的双向LSTM和CNN拼接 以及CRF预测 中文分词结果


## 文件说明
|序号|文件名|主要内容|
|---|----|--------|
|1. |Model.py|建立BiLSTM_CNN+CRF模型|
|2. |tools.py|存放所有需要用到函数方法|
|3. |parameter.py|放置所有参数|
|4. |train.py|训练模型|
|5. |predict.py|预测模型|


## 数据解释
|序号|文件名|作用|备注|
|----|---|----|---|
|1. |WordSeg.txt|用作训练使用|
|2. |test.txt|测试集|
|3. |eva.txt|预测集
|4. |pku_training.utf8|icws2中pku的标准训练预料|用来测试icws2 中的结果
|5. |pku_test_\[1-3\]|icws2中pku的预测预料|用来预测模型分割该预料的结果
|6. |pku_self.utf8|根据预测料分词后的效果|用作评测结果的输入
|7. |pku_train_politics.utf8|样本迁移后的中文分词训练语料库
|8. |pku_test_politics.utf8|用来预测政治领域分词
|9. |pku_gold_politics.utf8|对8.中政治领域分词的正确分词结果| 使用icwb2中的score测评

## 数据集和预测指令
1. 下载icws2的语料库：[icws2预料下载地址](http://sighan.cs.uchicago.edu/bakeoff2005/)
2. 进入icws2文件的script文件，输入一下指令:  
`./score ../gold/pku_training_words.utf8 ../gold/pku_test_gold.utf8 pku_self.utf8 > score.txt`  
详细用法可参考 [52nlp](http://www.52nlp.cn/%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E5%85%A5%E9%97%A8%E4%B9%8B%E8%B5%84%E6%BA%90)
3. 从score.txt中可以看到 RPF值

## 使用环境
- pyCharm 2017 professional
- python3.6
- tensorflow1.8
- MacOS Mojave

## 待完善
由于经验不足，代码中仍然存在不少bug，欢迎issue

