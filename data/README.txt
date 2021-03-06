* Introduction
本目录包含PKU和MSR这两个数据集的训练集、测试集和参考答案数据；同时也包含了评测脚本。
数据集的情况如下:
--------------------------------------------------------------------------------
Corpus             Encoding         Word        Words     Character   Characters
                                    Types                   Types
--------------------------------------------------------------------------------
Peking University  CP936            55,303    1,109,947     4,698      1,826,448
Microsoft Research CP936            88,119    2,368,391     5,167      4,050,469
--------------------------------------------------------------------------------

* File List
gold/       包含测试集和训练集的参考答案;
scripts/    包含评测脚本;
testing/    包含未被分词的测试集数据;
training/   包含未被分词的训练集数据;
     
* Encoding Issues
扩展名为 ".utf8" 的编码方式为 UTF-8 Unicode;
扩展名为 ".txt" 编码方式如下:
msr_   EUC-CN (CP936)
pku_   EUC-CN (CP936)

PS: EUC-CN 也叫做 "GB" 或者 "GB2312"。

* Scoring
 'score' 脚本的评价指标包含准确率、召回率和F值, 同时也会报告in-vocabulary和out-of-vocabulary的情况。脚本参数如下:
1. 训练集词表
2. 分词的标准答案
3. 分好词的测试集文件

参考方式如下：
% perl scripts/score gold/pku_training_words.utf8 gold/pku_test_gold.utf8 test_segmentation.utf8 > score.utf8
