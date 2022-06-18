# chinese_word_segmentation

## 项目介绍
以统计方法、神经网络方法、预训练语言模型方法为三条主线实现若干个中文分词模型，在PKU和MSR两个中文语料数据集上进行分词评测，并尝试进行一定的分析和改进

## 实现的分词模型

### 1. 统计机器学习方法
- HMM
- N-Gram
- JieBa* _(只做默认条件下的jieba库分词能力测试, 实现测试代码, 不进行训练)_

### 2. 神经网络方法
- CNN
- BiLSTM
- Transformer

### 3. 预训练方法
- PreTrained Character Embedding
- BERT
- WMSEG* _(直接引用论文结果来和本实验做对比, 不实现)_

### 其他技术
- 人工规则
- 分类器: CRF/Softmax

## 数据集

[SIGHAN Bakeoff 2005](http://sighan.cs.uchicago.edu/bakeoff2005/)官方数据集PKU和MSR

## 评测指标

- 精准率（Precision）：又称查准率，表示预测为正的样本中真实为正的样本占比
- 召回率（Recall）：又称查全率，表示真实为正的样本中预测为正的样本占比
- F值（F-measure）：又称F分数，是精确率和召回率的调和平均值
- 未登录词召回率（OOV Recall）：表示重复词区间未在词典中出现的词与标准分词中未在词典中出现的词的比值
- 登录词召回率（IV Recall）：表示重复词区间在词典中出现的词与标准分词在词典中出现的词的比值
- 推理速度（Inference Speed）：表示模型在测试集上做推理时平均每秒处理多少个字符（含单字和标点），单位为char/s

## 末期实验结果（CNN、BiLSTM、Transformer、Bert, 共10组实验）

本次实验中，我们组分别使用CNN、BiLSTM、Transformer、Bert四种Encoder对中文分词任务进行了建模，并在PKU数据集和MSR数据集上做了预测。实验设计及结果如下：
![](results/final%20results.png)

- 注：其中CNN+BiLSTM使用并行网络结构实现，而非串行结构

## 中期实验结果（HMM、N-gram、jieba, 共13组实验）

本次实验中，我们组分别使用HMM和N-gram两种模型对中文分词任务进行了建模，并在PKU数据集和MSR数据集上做了预测。尝试加入了人工规则，并对各种参数做了精调。实验设计及结果如下：
![](results/mid%20results.png)

## 说明

1. WMSEG是一种CWS任务训练框架，支持使用BiLSTM、BERT、ZEN作为encoder（ZEN和BERT一样也是一种预训练encoder，不过在中文语料上效果更好）。

## 参考
- [watermelon-lee的HMM实现](https://github.com/watermelon-lee/machine-learning-algorithms-implemented-by-python/tree/master/HMM)
- [基于N-gram的双向最大匹配中文分词](https://mqsee.blog.csdn.net/article/details/53466043)
- [另一位同学的课程项目实现](https://github.com/JackHCC/Chinese-Tokenization)
- [CNN+BiLSTM在CWS任务上的实现](https://github.com/FanhuaandLuomu/BiLstm_CNN_CRF_CWS)
- Transformer在CWS任务上的实现：[中文分词准SOTA: Unified Model (EMNLP2020)](https://aclanthology.org/2020.findings-emnlp.260/)
- BiLSTM/Bert在CWS任务上的实现：[代码](https://github.com/hemingkx/WordSeg) [知乎](https://zhuanlan.zhihu.com/p/371842740) [预训练字向量](https://github.com/Embedding/Chinese-Word-Vectors)
- [中文分词SOTA: WMSEG (ACL2020)](https://aclanthology.org/2020.acl-main.734/)
