# chinese_word_segmentation

## 项目介绍
以统计方法、神经网络方法、预训练语言模型方法为三条主线实现若干个中文分词模型，在PKU和MSR两个中文语料数据集上进行分词评测，并尝试进行一定的分析和改进

## 实现的分词模型

### 1. 统计机器学习方法
- N-Gram
- HMM
- MEMM (待实现)
- CRF (待实现)
- JieBa* _(只做默认条件下的jieba库分词能力测试，不进行训练)_

### 2. 神经网络方法
- CNN (待实现)
- BiLSTM (待实现)
- Transformer (待实现)

### 3. 预训练语言模型方法
- BERT (待实现)
- WMSEG (待实现)

## 数据集

[SIGHAN Bakeoff 2005](http://sighan.cs.uchicago.edu/bakeoff2005/)官方数据集PKU和MSR

## 评测指标

- 精准率（Precision）：又称查准率，表示预测为正的样本中真实为正的样本占比
- 召回率（Recall）：又称查全率，表示真实为正的样本中预测为正的样本占比
- F值（F-measure）：又称F分数，是精确率和召回率的调和平均值
- 未登录词召回率（OOV Recall）：表示重复词区间未在词典中出现的词与标准分词中未在词典中出现的词的比值
- 登录词召回率（IV Recall）：表示重复词区间在词典中出现的词与标准分词在词典中出现的词的比值
- 推理速度（Inference Speed）：表示模型在测试集上做推理时平均每秒处理多少个字符（含单字和标点），单位为char/s

## 中期实验结果（HMM、N-gram、jieba）

本次实验中，我们组分别使用HMM和N-gram两种模型对中文分词任务进行了建模，并在PKU数据集和MSR数据集上做了预测。然而不论是HMM还是N-gram模型，直接预测的效果均不是很好（传统模型的预测效果有限）。因此，我们尝试用基于规则的方法对N-gram模型进行了改进，并对各种参数做了精调。以下是各组实验的结果：

### 1. HMM参数实验结果

下表为HMM模型使用+δ平滑的参数实验结果。本次参数实验主要对HMM在PKU和MSR两个数据集上共2种场景下+δ平滑中的δ值进行了精调，分别得到了每个场景下的最优δ值。

![](https://github.com/SeaEagleI/chinese_word_segmentation/blob/master/results/1.png)

结论：
- HMM模型的推理速度适中，但效果较差。

可能的原因：
- 隐式马尔可夫模型是基于字标注的模型，它没能很好的利用词典的信息进行标注。
- 隐式马尔可夫模型对于其输出矩阵，转移矩阵，初始矩阵的空值较为敏感，简单的平滑处理不能很好的体现真实情况。
- 训练语料与测试中的句子长度也可能是其表现较差的原因。

### 2. N-Gram双向最大匹配参数实验结果

下表为在N-Gram（n=1,2,3）条件下使用双向最大匹配算法（即**匹配方式**=prepost）和+δ平滑，且不加入任何人工规则的参数实验结果。本次参数实验主要对n=1,2,3三种N-Gram在PKU和MSR两个数据集上共6种场景下+δ平滑中的δ值进行了精调。

![](https://github.com/SeaEagleI/chinese_word_segmentation/blob/master/results/2.png)

结论：
- 无规则情况下使用双向最大匹配的BiGram效果最好，在两个数据集上的F值均最高。

### 3. UniGram全切分混合参数实验结果

下表为在UniGram（即n=1）条件下对**匹配方式**、是否使用**外部规则**和**内部规则**三组参数进行的实验测试结果。

其中**匹配方式**共三种：双向最大匹配（prepost）、双向最大匹配和全切分算法混合匹配（hybrid）、全切分算法匹配（all-cut）。**混合匹配**算法是指先进行双向最大匹配，并对前向和后向匹配得到的分词结果不一致的子句进行全切分，从而避免了直接对整句进行全切分的高计算复杂度开销。

**外部规则**是指是否在测试集数据预处理阶段加入人工规则，从而将年份、日期和万、亿等数量词提前分开作为最终结果中的分词。

**全切分规则**亦即**内部规则**，和外部规则类似，不过是在全切分概率计算时对日期和数量词等赋一个较高的权重，从而让模型在分词时便于对这些难样本做出正确切分。

注：混合匹配和全切分匹配的实现均未使用+δ平滑，故这里不涉及δ值的设置和测试。

![](https://github.com/SeaEagleI/chinese_word_segmentation/blob/master/results/3.png)

结论：
- 混合方法在两个数据集上的最好表现均超过全切分，且速度上没有比双向最大匹配慢太多，整体上来看是三种算法中最优的。
- 基于规则的方法对模型在PKU数据集上的表现提升比MSR数据集要显著得多。(PKU数据集中日期和数量词较多，MSR数据集中日期和数量词很少，且模型在PKU训练集上不能很好地学到日期和数量词的正确分词模式，故针对日期和量词的人工规则对前者更有效)
- 全切分算法计算复杂度最高，其在匹配过程中的参与度越高，模型在测试集上的推理就越慢。
- 混合方法暴露给全切分算法的子句其实都不包含日期和数量词，所以是否使用全切分规则对混合算法的表现没有任何影响。

### 4. 多模型效果实验结果

最后，我们对比了每个模型在所有设置下的最优结果，同时引入了jieba、hannlp等外部库的实验结果作为参考。其中外部库不在PKU和MSR的训练集上做训练，直接在测试集上进行预测。

![](https://github.com/SeaEagleI/chinese_word_segmentation/blob/master/results/4.png)

结论：
- PKU上加入规则的UniGram表现最好，MSR上不加规则的BiGram表现最好。
- 由于HMM模型自身假设的限制，N-gram模型在PKU和MSR数据集上的预测表现均好于HMM模型。
- 相同模型在不同训练语料库上的表现不同，同一模型在MSR上的表现总要好于PKU。
- 外部库jieba由于不在PKU和MSR上训练，故效果较差。

## 说明

1. WMSEG是一种CWS任务训练框架，支持使用BiLSTM、BERT、ZEN作为encoder（ZEN和BERT一样也是一种预训练encoder，不过在中文语料上效果更好）。

## 参考
- [watermelon-lee的HMM实现](https://github.com/watermelon-lee/machine-learning-algorithms-implemented-by-python/tree/master/HMM)
- [基于N-gram的双向最大匹配中文分词](https://mqsee.blog.csdn.net/article/details/53466043)
- [另一位同学的课程项目实现（同步更新）](https://github.com/JackHCC/Chinese-Tokenization)
- [Bert/RoBerta/BiLSTM在CWS任务上的实现](https://github.com/hemingkx/WordSeg)
- [中文分词新SOTA: WMSEG (ACL2020)](https://aclanthology.org/2020.acl-main.734/)
