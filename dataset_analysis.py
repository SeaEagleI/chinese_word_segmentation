import os
import collections
from config import train_dir
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def tokenize(lines):
    """lines是多行列表，每一个元素是一行文本
    :returns List[List[str]]
    """
    return [line.strip().split() for line in lines]
def count_token(tokens):
    """tokens 是双重列表，每个元素是一个词列表
    :param tokens: List[List[str]]
    :returns dict{str: int}
    """
    tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
def sentence_length(text):
    """
    :param text: List[List[str]]
    :return: List[int]
    """
    return [len(sentence) for sentence in text]
def trainset_to_dict(filename):
    """将训练集转换为词典
    :returns vocab: dict{str: int}, sent_len: List[int]
    """
    vocab = None
    sent_len = None
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        tokens = tokenize(lines)
        sent_len = sentence_length(tokens)
        vocab = count_token(tokens)
    return dict(sorted(vocab.items(), key = lambda item: item[1], reverse = True)), sent_len

def show_len_sentence(sent_len, name):
    plt.plot([i for i in range(len(sent_len))],sorted(sent_len),'b-')
    plt.title(f'{name} sentence length')
    # 设置颜色为白色，隐藏横坐标标签
    # plt.xticks(color='w')
    plt.xlabel('sentence rank')
    plt.ylabel('length')
    plt.legend()
    plt.show()
def zipf_graph(vocab, name):
    """绘图验证zipf law"""
    freqs= [math.log10(freq) for _, freq in vocab.items()]
    token_rank = [math.log10(i) if i > 0 else 0 for i in range(len(freqs))]

    plt.plot(token_rank,freqs,'b-')
    plt.title(f'The zipf\'s law in {name} train set')
    # 设置颜色为白色，隐藏横坐标标签
    # plt.xticks(color='w')
    plt.xlabel('log(token)')
    plt.ylabel('log(frequency)')
    plt.legend()
    plt.show()

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def show_top_chinese(vocab, n):
    """绘制柱状图，显示前n个汉语高频词"""
    tokens = [token for token, _ in vocab.items()]
    freqs = [freq for _, freq in vocab.items()]

    chinese_token = [token for token in tokens if is_chinese(token) ]
    chinese_freq = [freqs[i] for i in range(len(tokens)) if is_chinese(tokens[i]) ]

    x_pos = [i for i in range(n)]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.bar(x_pos, chinese_freq[:n], align='center', alpha=0.7)
    plt.xticks(x_pos, chinese_token[:n]) # 设置x轴显示的内容

    for i in range(n):
        plt.text(x_pos[i] - 0.25, chinese_freq[i] + 0.5, str(chinese_freq[i]))
        # plt.text(x, y, str) 表示将文字内容str放在图表中，其中文字的左下角位于(x, y)处。如果需要居中，可以通过加减一个叫嚣数字进行偏移

    plt.xlabel("Chinese token")
    plt.ylabel('frequency')
    plt.title(f'Top {n} Chinese token')

    plt.show()



if __name__ == '__main__':

    root = ""
    pku_file, msr_file = os.path.join(root, f"{train_dir}/pku_training.utf8"), \
        os.path.join(root, f"{train_dir}/msr_training.utf8")
    vocab_pku, pku_sent_len = trainset_to_dict(pku_file)
    vocab_msr, msr_sent_len = trainset_to_dict(msr_file)
    print(len(vocab_pku), len(vocab_msr))
    print(len(pku_sent_len), len(msr_sent_len))
    zipf_graph(vocab_pku, "pku")
    zipf_graph(vocab_msr, "msr")
    show_top_chinese(vocab_pku, 20)
    show_top_chinese(vocab_msr, 20)
    show_len_sentence(pku_sent_len, "pku")
    show_len_sentence(msr_sent_len, "msr")
