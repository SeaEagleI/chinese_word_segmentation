# 工具箱，提供搭建模型的各个方法
import pickle
import re

import parameter as pm
import numpy as np
import keras as kr


def sentence2word_list(sentence):
    """
    作为get_words_content_label()子方法
    :param sentence: 类似于 '戈壁 上 ， 也 并不是 寸草不生'
    :return: 所有字符形成的list
    """
    word_list = []
    s = ''.join(sentence.split(' '))
    for c in s:
        word_list.append(c)
    return word_list


def sentence2label(sentence):
    """
    作为get_words_content_label()子方法
    :param sentence: 类似于 '戈壁 上 ， 也 并不是 寸草不生'
    :return: ['B','E','S','S','S','B','M','E','B','M','M','E']
    """
    label = []
    sentence = re.sub('   ', ' ', sentence) # 将三空格 转变为单空格
    sentence = re.sub('  ', ' ', sentence)  # 将双空格 转变为单空格
    l = sentence.split(' ')
    for item in l:
        if len(item) == 1:
            label.append('S')
        elif len(item) == 2:
            label.append('B')
            label.append('E')
        else:
            label.append('B')
            label.extend('M' * (len(item) - 2))
            label.append('E')
    return label


def get_words_content_label(filename):
    """
    将文本转化为 文字集合words, 文本内容列表, 文本内容列表每个字对应标签
    文字集合 set('西'，'瓜'，'苹'，'果'，'真'，'甜')
    文本类容列表[['西'，'瓜'，'真'，'甜']，['苹'，'果'，'真'，'香']]
    标签       [['B', 'M', 'M',  'E'], ['B',  'M', 'M', 'E']]
    :param filename: 需要提取的文件名称
    :return: 返回 如上三项
    """
    words_list, content, label = [], [], []
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.readlines()
    for line in text:
        line = line.strip()
        cur_words_list = sentence2word_list(line)
        words_list.extend(cur_words_list)
        content.append(cur_words_list)
        label.append(sentence2label(line))

    return set(words_list), content, label


def decode_word(words_set):
    """
    字符集合编码器 的字典
    :param words_set:字符集合
    :return: 字典
    """
    dictionary = {'<PAD>': 0, '<UNK>': 1}
    index = 2
    for c in words_set:
        dictionary[c] = index
        index += 1
    with open('./data/dictionary.pkl', 'wb') as fw:
        pickle.dump(dictionary, fw)
    return dictionary


def content2id(dictionary, content):
    """
    对content 依据 dictionary进行编码
    :param dictionary: 字典
    :param content: 由文本提取出的content
    :return: 返回编码后的content_id
    """
    content_id = []
    for s in content:
        tmp = []
        for c in s:
            if c not in dictionary:
                c = '<UNK>'
            tmp.append(dictionary[c])
        content_id.append(tmp)
    return content_id


def label2id(label):
    """
    对label [BMES] 进行编码，分别对应[0，1，2，3]
    :param label: 由文本提取出的标签
    :return: 返回编码后的标签label_id
    """
    dictionary = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    label_id = []
    for s in label:
        label_id.append([dictionary[x] for x in s])
    return label_id


def get_dictionary():
    """
    获得  保存的pkl字典
    :return:
    """
    with open('./data/dictionary.pkl', 'rb') as fr:
        dictionary = pickle.load(fr)
    return dictionary



# ---------------------------------------------------- #

def next_batch(x, y, batch_size=pm.batch_size):
    """
    生成器
    返回数据各个批次 的x 和 label
    :param x:
    :param y:
    :param batch_size:
    :return:
    """
    l = len(x)
    x = np.array(x)
    y = np.array(y)
    num_batch = int((l - 1) / batch_size) + 1
    # 打乱 原序列顺序
    indices = np.random.permutation(l)
    x_permutation = x[indices]
    y_permutation = y[indices]

    for i in range(num_batch):
        start = i * batch_size
        end = min((i + 1) * batch_size, l)
        yield x_permutation[start:end], y_permutation[start:end]


def sentence_process(x_batch):
    """
    将x_batch中所有句子补齐到最大长度，末尾用0补齐
    返回该批次每个句子原本长度
    :param x_batch: [[句子],[句子],[句子]，..., [句子]]
    :return: 补齐后的句子 x_batch_with_pad, 该批次每个句子原本长度
    """
    ori_length = []
    # max_length = max(map(lambda x: len(x), x_batch))
    max_length = pm.seq_max_len
    for s in x_batch:
        ori_length.append(len(s))
    x_batch_with_pad = kr.preprocessing.sequence.pad_sequences(x_batch, max_length, padding='post', truncating='post')

    return x_batch_with_pad, ori_length

