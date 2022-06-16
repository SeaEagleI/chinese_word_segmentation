# 使用transformer encoder + crf / softmax进行分词

根据https://github.com/acphile/MCCWS进行修改


## 训练过程
* 需要自行下载sighan2005数据集，放在data文件下

* 预处理原始数据集，将原始数据集处理为 character label 的 pair 集合
```
python prepoccess.py --dataset <dataset name> --output <output directionary name>
```

* 为分词模型生成字典与数据集
```
python makedict.py --datasets <dataset name>
python make_dataset.py --training-data data/<dataset name>/bmes/train-all.txt --test-data data/<dataset name>/bmes/test.txt -o <output_path>
```

生成的.pkl文件格式如下
```
{
    'train_set': fastNLP.DataSet
    'test_set': fastNLP.DataSet
    'uni_vocab': fastNLP.Vocabulary, vocabulary of unigram
    'bi_vocab': fastNLP.Vocabulary, vocabulary of bigram
    'tag_vocab': fastNLP.Vocabulary, vocabulary of BIES
    'task_vocab': fastNLP.Vocabulary, vocabulary of criteria
}
```

* 最后，使用下列命令进行训练。这种模式下，不采用预训练的词向量，也不采用bigram embedding
```
python main.py --dataset <output_path> --crf --devi <num>
```

* 具体命令详见main.py文件:
```
python main.py --help
```
* 训练结果在result目录下