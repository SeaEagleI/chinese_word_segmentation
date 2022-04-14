# -*- coding:utf-8 -*-
import math
from config import train_dir, test_dir, pred_dir
from evaluate import eval

# model params
model_type = "ngram"
span = 12
Punctuation = [u'、', u'”', u'“', u'。', u'（', u'）', u'：', u'《', u'》', u'；', u'！', u'，', u'、']
Number = [u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'%', u'.']
English = [u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'q', u'r',
           u's', u't', u'u', u'v', u'w', u'x', u'y', u'z', u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J',
           u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R', u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'Z']


# 前向后向最大匹配算法实现
class PrePostNgram:
    def __init__(self, train_file, test_file, pred_file):
        self._WordDict = {}
        self._NextCount = {}
        self._NextSize = 0
        self._WordSize = 0
        self.train_file = train_file
        self.test_file = test_file
        self.pred_file = pred_file

    def Training(self):
        """
        读取训练集文件
        得到每个词出现的个数 self._WordDict
        得到每个词后接词出现的个数 self._NextCount
        :return:
        """
        print('start training...')
        self._NextCount[u'<BEG>'] = {}
        traing_file = open(self.train_file, encoding="utf-8")
        traing_cnt = 0
        for line in traing_file:
            line = line.strip()
            line = line.split(' ')
            line_list = []
            # 得到每个词出现的个数
            for pos, words in enumerate(line):
                if words != u'' and words not in Punctuation:
                    line_list.append(words)
            traing_cnt += len(line_list)
            for pos, words in enumerate(line_list):
                if not words in self._WordDict:
                    self._WordDict[words] = 1
                else:
                    self._WordDict[words] += 1
                # 得到每个词后接词出现的个数
                words1, words2 = u'', u''
                if pos == 0:
                    words1, words2 = u'<BEG>', words
                elif pos == len(line_list) - 1:
                    words1, words2 = words, u'<END>'
                else:
                    words1, words2 = words, line_list[pos + 1]
                if not words1 in self._NextCount:
                    self._NextCount[words1] = {}
                if not words2 in self._NextCount[words1]:
                    self._NextCount[words1][words2] = 1
                else:
                    self._NextCount[words1][words2] += 1

        traing_file.close()
        self._NextSize = traing_cnt
        print('total training words length is: ', traing_cnt)
        print('training done...')
        self._WordSize = len(self._WordDict)
        print("len _WordDict: ", len(self._WordDict))
        print("len _NextCount: ", len(self._NextCount))

    # $mode在"Pre"、"Post"、"prepostBigram"中选择，分别表示前向匹配、后向匹配、双向匹配
    # 双向匹配在pku和msr两个数据集上效果均比前两种单向匹配方法F值高一个点左右
    def SeparWords(self, mode):
        print('start SeparWords...')
        test_file = open(self.test_file, encoding="utf-8")
        test_result_file = open(self.pred_file, 'w', encoding="utf-8")

        SenListCnt = 0
        tmp_words = u''
        SpecialDict = {}
        for line in test_file:
            # 编码方式改为utf-8
            line = line.strip()
            SenList = []

            # 根据英文、数字、标点将长句切分为多个子句
            flag = 0
            for ch in line:
                if ch in Number or ch in English:
                    flag = 1
                    tmp_words += ch
                # bug fix: 修正对长度为1、只有一个标点符号的句子的识别错误（例如MSR Line 2969）
                elif ch in Punctuation:
                    if tmp_words != u'':
                        SenList.append(tmp_words)
                        SenListCnt += 1
                        if flag == 1:
                            SpecialDict[tmp_words] = 1
                            flag = 0
                    SenList.append(ch)
                    tmp_words = u''
                else:
                    if flag == 1:
                        SenList.append(tmp_words)
                        SenListCnt += 1
                        SpecialDict[tmp_words] = 1
                        flag = 0
                        tmp_words = ch
                    else:
                        tmp_words += ch
            if tmp_words != u'':
                SenList.append(tmp_words)
                SenListCnt += 1
                if flag == 1:
                    SpecialDict[tmp_words] = 1
            tmp_words = u''

            for ch in SenList:
                if ch not in Punctuation and ch not in SpecialDict:
                    if mode == 'Pre':
                        ParseList = self.PreMax(ch)
                    elif mode == 'Post':
                        ParseList = self.PosMax(ch)
                    else:
                        ParseList1 = self.PreMax(ch)
                        ParseList2 = self.PosMax(ch)
                        ParseList1.insert(0, u'<BEG>')
                        ParseList1.append(u'<END>')
                        ParseList2.insert(0, u'<BEG>')
                        ParseList2.append(u'<END>')
                        # 根据前向最大匹配和后向最大匹配得到得到句子的两个词序列（添加BEG和END作为句子的开始和结束）

                        # 记录最终选择后拼接得到的句子
                        ParseList = []

                        # CalList1和CalList2分别记录两个句子词序列不同的部分
                        CalList1 = []
                        CalList2 = []

                        # pos1和pos2记录两个句子的当前字的位置，cur1和cur2记录两个句子的第几个词
                        pos1 = pos2 = 0
                        cur1 = cur2 = 0
                        while True:
                            if cur1 == len(ParseList1) and cur2 == len(ParseList2):
                                break
                            # 如果当前位置一样
                            if pos1 == pos2:
                                # 当前位置一样，并且词也一样
                                if len(ParseList1[cur1]) == len(ParseList2[cur2]):
                                    pos1 += len(ParseList1[cur1])
                                    pos2 += len(ParseList2[cur2])
                                    # 说明此时得到两个不同的词序列，根据bigram选择概率大的
                                    # 注意算不同的时候要考虑加上前面一个词和后面一个词，拼接的时候再去掉即可
                                    if len(CalList1) > 0:
                                        CalList1.insert(0, ParseList[-1])
                                        CalList2.insert(0, ParseList[-1])
                                        if cur1 < len(ParseList1) - 1:
                                            CalList1.append(ParseList1[cur1])
                                            CalList2.append(ParseList2[cur2])

                                        p1 = self.CalSegProbability(CalList1)
                                        p2 = self.CalSegProbability(CalList2)
                                        if p1 > p2:
                                            CalList = CalList1
                                        else:
                                            CalList = CalList2
                                        CalList.remove(CalList[0])
                                        if cur1 < len(ParseList1) - 1:
                                            CalList.remove(ParseList1[cur1])
                                        for words in CalList:
                                            ParseList.append(words)
                                        CalList1 = []
                                        CalList2 = []
                                    ParseList.append(ParseList1[cur1])
                                    cur1 += 1
                                    cur2 += 1
                                # pos1相同，len(ParseList1[cur1])不同，向后滑动，不同的添加到list中
                                elif len(ParseList1[cur1]) > len(ParseList2[cur2]):
                                    CalList2.append(ParseList2[cur2])
                                    pos2 += len(ParseList2[cur2])
                                    cur2 += 1
                                else:
                                    CalList1.append(ParseList1[cur1])
                                    pos1 += len(ParseList1[cur1])
                                    cur1 += 1
                            else:
                                # pos1不同，而结束的位置相同，两个同时向后滑动
                                if pos1 + len(ParseList1[cur1]) == pos2 + len(ParseList2[cur2]):
                                    CalList1.append(ParseList1[cur1])
                                    CalList2.append(ParseList2[cur2])
                                    pos1 += len(ParseList1[cur1])
                                    pos2 += len(ParseList2[cur2])
                                    cur1 += 1
                                    cur2 += 1
                                elif pos1 + len(ParseList1[cur1]) > pos2 + len(ParseList2[cur2]):
                                    CalList2.append(ParseList2[cur2])
                                    pos2 += len(ParseList2[cur2])
                                    cur2 += 1
                                else:
                                    CalList1.append(ParseList1[cur1])
                                    pos1 += len(ParseList1[cur1])
                                    cur1 += 1
                        ParseList.remove(u'<BEG>')
                        ParseList.remove(u'<END>')

                    for pos, words in enumerate(ParseList):
                        tmp_words += words + u'  '
                else:
                    tmp_words += ch + u'  '
            test_result_file.write(tmp_words + '\n')
            tmp_words = u''

        test_file.close()
        test_result_file.close()
        print('SenList length: ', SenListCnt)
        print('SeparWords done...')

    def CalSegProbability(self, ParseList):
        p = 0
        # 由于概率很小，对连乘做了取对数处理转化为加法
        for pos, words in enumerate(ParseList):
            if pos < len(ParseList) - 1:
                # 乘以后面词的条件概率
                word1, word2 = words, ParseList[pos + 1]
                if not  word1 in self._NextCount:
                    # 加1平滑
                    p += math.log(1.0 / self._NextSize)
                else:
                    # 加1平滑
                    fenzi, fenmu = 1.0, self._NextSize
                    for key in self._NextCount[word1]:
                        if key == word2:
                            fenzi += self._NextCount[word1][word2]
                        fenmu += self._NextCount[word1][key]
                    p += math.log((fenzi / fenmu))
            # 乘以第一个词的概率
            if (pos == 0 and words != u'<BEG>') or (pos == 1 and ParseList[0] == u'<BEG>'):
                if words in self._WordDict:
                    p += math.log(float(self._WordDict[words]) + 1 / self._WordSize + self._NextSize)
                else:
                    # 加1平滑
                    p += math.log(1 / self._WordSize + self._NextSize)
        return p

    def PreMax(self, sentence):
        """
        把每个句子正向最大匹配
        """
        cur, tail = 0, span
        ParseList = []
        while (cur < tail and cur <= len(sentence)):
            if len(sentence) < tail:
                tail = len(sentence)
            if tail == cur + 1:
                ParseList.append(sentence[cur:tail])
                cur += 1
                tail = cur + span
            elif sentence[cur:tail] in self._WordDict:
                ParseList.append(sentence[cur:tail])
                cur = tail
                tail = cur + span
            else:
                tail -= 1
        return ParseList

    def PosMax(self, sentence):
        """
        把每个句子后向最大匹配
        :param sentence:
        :return:
        """
        cur = len(sentence) - span
        tail = len(sentence)
        if cur < 0:
            cur = 0

        ParseList = []
        while (cur < tail and tail > 0):
            if tail == cur + 1:
                ParseList.append(sentence[cur:tail])
                tail -= 1
                cur = tail - span
                if cur < 0:
                    cur = 0
            elif sentence[cur:tail] in self._WordDict:
                ParseList.append(sentence[cur:tail])
                tail = cur
                cur = tail - span
                if cur < 0:
                    cur = 0
            else:
                cur += 1
        ParseList.reverse()
        return ParseList


# train & test loop
def train_and_test(dataset="pku"):
    # data files
    train_file = f"{train_dir}/{dataset}_training.utf8"
    test_file = f"{test_dir}/{dataset}_test.utf8"
    pred_file = f"{pred_dir}/{dataset}_test_pred_{model_type}.utf8"
    gold_file = f"{test_dir}/{dataset}_test_gold.utf8"

    # train
    p = PrePostNgram(train_file, test_file, pred_file)
    p.Training()
    # test
    p.SeparWords('prepostBigram')
    print('PrePostSegBigram Max')
    # eval
    eval(pred_file, gold_file)


if __name__ == '__main__':

    train_and_test("pku")
    train_and_test("msr")

