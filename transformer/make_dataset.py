import os
import sys

import codecs
import argparse
import pickle
import collections
from utils import get_processing_word, is_dataset_tag, make_sure_path_exists
from fastNLP import Instance, DataSet, Vocabulary, Const

def expand(x):
    sent=["<sos>"]+x[1:]+["<eos>"]
    return [x+y for x,y in zip(sent[:-1],sent[1:])]
    
def read_file(filename, processing_word=get_processing_word(lowercase=False)):
    dataset = DataSet()
    niter=0
    line_num = 0
    with codecs.open(filename, "r", "utf-16") as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(words) != 0:
                    assert len(words)>2, f"line {line_num}: line={line} and len(words)={len(words)} and words = {words}"
                    if niter==1:
                        print(words,tags)
                    niter += 1
                    dataset.append(Instance(ori_words=words[:-1], ori_tags=tags[:-1]))
                    words, tags = [], []
            else:
                word, tag = line.split()
                word = processing_word(word)
                words.append(word)
                tags.append(tag.lower())
            line_num += 1
    dataset.apply_field(lambda x: [x[0]], field_name='ori_words', new_field_name='task')   
    dataset.apply_field(lambda x: len(x), field_name='ori_tags', new_field_name='seq_len')   
    dataset.apply_field(lambda x: expand(x), field_name='ori_words', new_field_name="bi1")
    return dataset


parser = argparse.ArgumentParser()
parser.add_argument("--training-data", required=True, dest="training_data", help="Training data .txt file")
parser.add_argument("--test-data", required=True, dest="test_data", help="Test data .txt file")
parser.add_argument("-o", required=True, dest="output", help="Output filename (.pkl)")

options = parser.parse_args()

print('Making training dataset')
train_set = read_file(options.training_data)
print('Making test dataset')
test_set = read_file(options.test_data)

uni_vocab = Vocabulary(min_freq=None).from_dataset(train_set,test_set, field_name='ori_words')
bi_vocab = Vocabulary(min_freq=3).from_dataset(train_set,test_set, field_name="bi1")
tag_vocab = Vocabulary(min_freq=None, padding="s", unknown=None).from_dataset(train_set, field_name='ori_tags')
task_vocab = Vocabulary(min_freq=None,padding=None, unknown=None).from_dataset(train_set, field_name='task')

def to_index(dataset):
    uni_vocab.index_dataset(dataset, field_name='ori_words',new_field_name='uni')
    tag_vocab.index_dataset(dataset, field_name='ori_tags',new_field_name='tags')
    task_vocab.index_dataset(dataset, field_name='task',new_field_name='task')
    
    dataset.apply_field(lambda x: x[1:], field_name='bi1', new_field_name="bi2")
    dataset.apply_field(lambda x: x[:-1], field_name='bi1', new_field_name="bi1")
    bi_vocab.index_dataset(dataset, field_name='bi1',new_field_name='bi1')
    bi_vocab.index_dataset(dataset, field_name='bi2',new_field_name='bi2')
    
    dataset.set_input("task","uni","bi1","bi2","seq_len")
    dataset.set_target("tags")
    return dataset
    
train_set = to_index(train_set)      
test_set = to_index(test_set)

output={}
output["train_set"]=train_set
output["test_set"]=test_set
output["uni_vocab"]=uni_vocab
output["bi_vocab"]=bi_vocab
output["tag_vocab"]=tag_vocab
output["task_vocab"]=task_vocab

make_sure_path_exists(os.path.dirname(options.output))

print('Saving dataset to {}'.format(options.output))
with open(options.output, "wb") as outfile:
    pickle.dump(output, outfile)

print(len(task_vocab),len(tag_vocab),len(uni_vocab),len(bi_vocab))
