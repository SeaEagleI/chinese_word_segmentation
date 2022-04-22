# chinese_word_segmentation

## 项目介绍
以统计方法、神经网络方法、预训练语言模型方法为三条主线实现若干个中文分词模型，在PKU和MSR两个中文语料数据集上进行分词评测，并尝试进行一定的分析和改进

## 实现的分词模型

### 统计机器学习方法
- N-Gram
- HMM
- MEMM (待实现)
- CRF (待实现)
- JieBa* _(只做默认条件下的jieba库分词能力测试，不进行训练)_

### 神经网络方法
- CNN (待实现)
- BiLSTM (待实现)
- Transformer (待实现)

### 预训练语言模型方法
- BERT (待实现)
- WMSEG (待实现)

## 评测结果

- 在PKU和MSR上的评测结果（F值）

（待更新）

## 说明

1. WMSEG是一种CWS任务训练框架，支持使用BiLSTM、BERT、ZEN作为encoder（ZEN和BERT一样也是一种预训练encoder，不过在中文语料上效果更好）。

## 参考
- [watermelon-lee的HMM实现](https://github.com/watermelon-lee/machine-learning-algorithms-implemented-by-python/tree/master/HMM)
- [基于N-gram的双向最大匹配中文分词](https://mqsee.blog.csdn.net/article/details/53466043)
- [另一位同学的课程项目实现（同步更新）](https://github.com/JackHCC/Chinese-Tokenization)
- [Bert/RoBerta/BiLSTM在CWS任务上的实现](https://github.com/hemingkx/WordSeg)
- [中文分词新SOTA: WMSEG (ACL2020)](https://aclanthology.org/2020.acl-main.734/)
