# chinese_word_segmentation

## 项目介绍
以统计方法和NN方法为两条主线实现若干个中文分词模型，在PKU和MSR两个中文语料数据集上进行分词评测，并尝试进行一定的分析和改进

## 实现的分词模型

### 统计机器学习方法
- HMM
- N-Gram
- JieBa (只做默认条件下的jieba库分词能力测试，不进行训练)

### 神经网络方法
- Bert (待实现)
- WMSEG (待实现)

## 评测结果

- 在PKU和MSR上的评测结果（F值）

|        |    PKU    |    MSR    |  Comment  |
|:------:|:---------:|:---------:|:---------:|
|  HMM   |   76.44   |   79.14   |           |
| JieBa  |   81.93   |   81.41   | w/o Train |
| N-Gram |   87.02   |   94.17   |           |
| WMSEG  | **96.53** | **98.40** |           |

## 参考
- [watermelon-lee的HMM实现](https://github.com/watermelon-lee/machine-learning-algorithms-implemented-by-python/tree/master/HMM)
- [基于N-gram的双向最大匹配中文分词](https://mqsee.blog.csdn.net/article/details/53466043)
- [结巴中文分词: 做最好的Python中文分词组件](https://github.com/fxsjy/jieba)
- [中文分词新SOTA: WMSEG (ACL2020)](https://aclanthology.org/2020.acl-main.734/)
