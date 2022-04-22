# -*- coding:utf-8 -*-
import math
from config import train_dir, test_dir, pred_dir
from evaluate import eval
import time

# model params
model_type = "ngram"
span = 12
Punctuation = u'、""“”’‘。（）():[]【】;：《》；！，、'
Number = u'0123456789%.％．@○'
English = u'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


# 前向后向最大匹配算法实现
class N_Gram:
    def __init__(self, train_file, test_file, pred_file):
        # stats params
        ## 1-gram params
        self._WordDict = {}  # Dict[w -> count]
        self._UniVocabSize = 0  # len(_WordDict)
        self._UniWordCount = 0  # sum(count)
        ## 2-gram params
        self._NextWord = {}  # Dict[w -> Dict[next_w -> count]]
        self._BiVocabSize = 0  # len(_NextWord)
        self._BiWordCount = {}  # Dict[w -> sum(count)]
        ## 3-gram params
        self._NextNextWord = {}  # Dict[w -> Dict[next_w -> Dict[next_next_w -> count]]]
        self._TriVocabSize = 0  # len(_NextNextWord)
        self._TriWordCount = {}  # Dict[w -> Dict[next_w -> sum(count)]]

        # data files
        self.train_file = train_file
        self.test_file = test_file
        self.pred_file = pred_file

        # funcs
        self.SegProbs = {
            1: self.UniSegProb,
            2: self.BiSegProb,
            3: self.TriSegProb,
        }
        self.inner_rules = True

    # 读取训练集文件，得到每个词的1元词频 _WordDict 和 每个词后接词的2元词频 _NextWord
    def Training(self):
        for line in open(self.train_file, encoding="utf-8"):
            # 对训练语料预处理，得到词切分序列
            line_list = []
            for word in line.strip().split(' '):
                if word != u'' and word not in Punctuation:
                    line_list.append(word)
            # 统计每个词的词频（1-gram）
            for pos, word in enumerate(line_list):
                self.add_word(word)
            # 统计每个词的后接词词频（2-gram）
            line_list = [u'<BEG>'] + line_list + [u'<END>']
            for pos, word in enumerate(line_list[:-1]):
                self.add_next_word(word, line_list[pos + 1])
            # 统计每个词的2阶后接词词频（3-gram）
            line_list = [u'<BEG>'] + line_list + [u'<END>']
            for pos, word in enumerate(line_list[:-2]):
                self.add_next_next_word(word, line_list[pos + 1], line_list[pos + 2])

        # set statistical variables
        self._UniVocabSize = len(self._WordDict)
        self._UniWordCount = sum(self._WordDict.values())
        self._BiVocabSize = len(self._NextWord)
        self._BiWordCount = {w: sum(d.values()) for w, d in self._NextWord.items()}
        self._TriVocabSize = len(self._NextNextWord)
        self._TriWordCount = {w1: {w2: sum(d2.values()) for w2, d2 in d1.items()} for w1, d1 in
                              self._NextNextWord.items()}
        # print('_UniVocabSize: ', self._UniVocabSize)
        # print("_UniWordCount: ", self._UniWordCount)

    # 将<word>组合计入_WordDict
    def add_word(self, word):
        if word not in self._WordDict:
            self._WordDict[word] = 1
        else:
            self._WordDict[word] += 1

    # 将<word1,word2>组合计入_NextWord
    def add_next_word(self, word1, word2):
        if word1 not in self._NextWord:
            self._NextWord[word1] = {}
        if word2 not in self._NextWord[word1]:
            self._NextWord[word1][word2] = 1
        else:
            self._NextWord[word1][word2] += 1

    # 将<word1,word2,word3>组合计入_NextNextWord
    def add_next_next_word(self, word1, word2, word3):
        if word1 not in self._NextNextWord:
            self._NextNextWord[word1] = {}
        if word2 not in self._NextNextWord[word1]:
            self._NextNextWord[word1][word2] = {}
        if word3 not in self._NextNextWord[word1][word2]:
            self._NextNextWord[word1][word2][word3] = 1
        else:
            self._NextNextWord[word1][word2][word3] += 1

    # 使用1-gram和规则做全切分
    def all_cut(self, sent):
        sent = sent.strip()
        log = lambda x: float('-inf') if not x else math.log(x)
        # freq = lambda x: self.dict[x] if x in self.dict else 0 if len(x)>1 else 1  # 计算每个词的频次（加入平滑）
        # Viterbi算法：从前向后打表计算到每个位置的最优上一个切分点
        # cut_point[i]表示到sent[i]的上一个最优分割点下标, max_prob[i]表示上述最优分割的1-gram累计概率
        l = len(sent)
        max_prob, cut_point = [0] * (l + 1), [0] * (l + 1)
        for i in range(1, l + 1):
            # 加规则
            max_prob[i], cut_point[i] = max([(log(self.freq(sent[k:i]) / self._UniVocabSize) + max_prob[k], k) for k in range(0, i)])
        # 从后向前回溯得到最优分词序列
        words, pos = [], l
        while pos != 0:
            last_pos = cut_point[pos]
            words.insert(0, sent[last_pos: pos])
            pos = last_pos
        return words

    # 计算1-gram下给定单切分片的概率
    def freq(self, x):
        if x in self._WordDict:
            return self._WordDict[x]
        # 符合日期和量词特征的返回1.34
        # if self.inner_rules and self.is_date_or_number(x) or self.is_special_string(x):
        if self.inner_rules and self.is_date_or_quantifier(x):
                return max(self._WordDict.values()) / self._UniVocabSize
        # 不是单字的没有概率返回0, 单字的不在词典的话返回1
        return int(len(x) == 1)

    # 判断是否为全字母、中英文数字、和典型数字符号组成的串
    def is_all_seperator(self, s):
        if not s:
            return False
        for ch in s:
            if not (ch in English or ch in Number):
                return False
        return True

    # 判断一个字符串是否为日期或者量词: s = <NUM> | <NUM>({年月日时分万亿}|万亿)
    def is_date_or_quantifier(self, s):
        end_s = '年月日时分万亿'
        # 不包含中文和℃的情况
        if self.is_all_seperator(s):
            return True
        # 手动加规则处理 以"年月日时分"结尾的日期 和 以"亿""万""万亿"结尾的数量单位
        if s[-1] in end_s and self.is_all_seperator(s[:-1]):
            return True
        elif s[-2:] == '万亿' and self.is_all_seperator(s[:-2]):
            return True
        else:
            return False

    # 在测试集上做分词
    # 双向匹配在pku和msr两个数据集上效果均比前两种单向匹配方法F值高一个点左右
    def Segmentation(self, n=1, delta=1, outer_rules=True, method="cross", inner_rules=True, log=False):
        # set file & func
        test_file = open(self.test_file, encoding="utf-8")
        test_result_file = open(self.pred_file, 'w', encoding="utf-8")
        CalSegProb = self.SegProbs[n]
        self.inner_rules = inner_rules
        assert method in ["prepost", "cross", "all-cut"]
        # if all_cut:
        #     assert n == 1
        # init vars
        start_t = time.perf_counter()
        char_count, sent_count = 0, 0
        tmp_words = u''
        SpecialDict = {}
        for line in test_file:
            # 编码方式改为utf-8
            line = line.strip()
            char_count += len(line)
            SenList, SenList1, SenList2 = [], [], []

            # 根据英文、数字、标点将长句切分为多个子句
            # 切<标点>
            for ch in line:
                if ch in Punctuation:
                    if tmp_words != u'':
                        SenList1.append(tmp_words)
                        sent_count += 1
                    SenList1.append(ch)
                    tmp_words = u''
                else:
                    tmp_words += ch
            if tmp_words != u'':
                SenList1.append(tmp_words)
                sent_count += 1
            tmp_words = u''
            # Outer_Rules: 在外部切<英文>、<数字>、以及<数字+量词>
            if outer_rules:
                for sent in SenList1:
                    if sent[0] in Punctuation:
                        SenList2.append(sent)
                        continue
                    flag = 0
                    for ch in sent:
                        if self.is_all_seperator(ch):
                            if flag == 0 and tmp_words != u'':
                                SenList2.append(tmp_words)
                                tmp_words = u''
                            flag = 1
                            tmp_words += ch
                        else:
                            if flag == 1:
                                if ch in "年月日时分亿":
                                    SenList2.append(tmp_words + ch)
                                    SpecialDict[tmp_words + ch] = 1
                                    flag = 0
                                    tmp_words = u''
                                elif ch in "万":
                                    tmp_words += ch
                                else:
                                    SenList2.append(tmp_words)
                                    SpecialDict[tmp_words] = 1
                                    flag = 0
                                    tmp_words = ch
                            else:
                                tmp_words += ch
                    if tmp_words != u'':
                        SenList2.append(tmp_words)
                        if flag == 1:
                            SpecialDict[tmp_words] = 1
                    tmp_words = u''
                SenList = SenList2
            else:
                SenList = SenList1

            # 逐句切分
            for sent in SenList:
                if sent[0] not in Punctuation and sent not in SpecialDict:
                    ParseList = []
                    # n-gram均可使用双向最大匹配 和 OuterRules (n=1,2,3)
                    # 1-gram可以使用全切分, 并在全切分内部选择是否使用 InnerRules (n=1)
                    if method == "all-cut":
                        ParseList = self.all_cut(sent)
                    else:
                        # 根据前向最大匹配和后向最大匹配得到得到句子的两个词序列（添加BEG和END作为句子的开始和结束）
                        begins, ends = [u'<BEG>'] * (n - 1), [u'<END>'] * (n - 1)
                        ParseList1 = begins + self.PreMax(sent) + ends
                        ParseList2 = begins + self.PostMax(sent) + ends
                        # ParseList记录最终分词结果, CalList1和CalList2分别记录两个句子词序列不同的部分
                        # pos1和pos2记录两个句子的当前字的位置，cur1和cur2记录两个句子的第几个词
                        CalList1, CalList2 = [], []
                        pos1, pos2, cur1, cur2 = 0, 0, 0, 0
                        while True:
                            if cur1 == len(ParseList1) and cur2 == len(ParseList2):
                                break
                            # posx一样，并且下一个词也一样
                            if pos1 == pos2 and len(ParseList1[cur1]) == len(ParseList2[cur2]):
                                pos1 += len(ParseList1[cur1])
                                pos2 += len(ParseList2[cur2])
                                # 说明此时得到两个不同的词序列，根据n-gram选择概率大的
                                if len(CalList1) > 0:
                                    # 算不同的时候视n的大小考虑加上前面n-1个词和后面n-1个词，拼接的时候再去掉即可
                                    if n == 2:
                                        # assert cur1 <= len(ParseList1) - 1
                                        CalList1 = ParseList[-1:] + CalList1 + ParseList1[cur1: cur1 + 1]
                                        CalList2 = ParseList[-1:] + CalList2 + ParseList2[cur2: cur2 + 1]
                                    elif n == 3:
                                        # assert cur1 <= len(ParseList1) - 2 and cur2 <= len(ParseList2) - 2
                                        CalList1 = ParseList[-2:] + CalList1 + ParseList1[cur1: cur1 + 2]
                                        CalList2 = ParseList[-2:] + CalList2 + ParseList2[cur2: cur2 + 2]
                                    # 全切分（1-gram, 只对双向匹配过程中不一致的部分做消歧, 从而降低全局计算量）
                                    if n == 1 and method == "cross":
                                        CalList = self.all_cut(''.join(CalList1))
                                        t = 1
                                    # 非全切分
                                    elif method == "prepost":
                                        # 计算序列概率，选择较大者
                                        p1 = CalSegProb(CalList1, delta)
                                        p2 = CalSegProb(CalList2, delta)
                                        CalList = CalList1 if p1 > p2 else CalList2
                                    # 将前面n-1个词和后面n-1个词去掉
                                    CalList = CalList[n - 1: 1 - n] if n >= 2 else CalList
                                    for word in CalList:
                                        ParseList.append(word)
                                    CalList1, CalList2 = [], []
                                ParseList.append(ParseList1[cur1])
                                cur1 += 1
                                cur2 += 1
                            # posx不同，而结束位相同，两个同时向后滑动
                            elif pos1 != pos2 and pos1 + len(ParseList1[cur1]) == pos2 + len(ParseList2[cur2]):
                                CalList1.append(ParseList1[cur1])
                                CalList2.append(ParseList2[cur2])
                                pos1 += len(ParseList1[cur1])
                                pos2 += len(ParseList2[cur2])
                                cur1 += 1
                                cur2 += 1
                            # posx相同，posx+len(ParseListx[curx])不同，将结束位靠前的向后滑动，并将词添加到待计算序列CalListx中
                            elif pos1 + len(ParseList1[cur1]) > pos2 + len(ParseList2[cur2]):
                                CalList2.append(ParseList2[cur2])
                                pos2 += len(ParseList2[cur2])
                                cur2 += 1
                            else:
                                CalList1.append(ParseList1[cur1])
                                pos1 += len(ParseList1[cur1])
                                cur1 += 1
                        # 将开头和末尾的<BEG>和<END>去掉
                        ParseList = ParseList[n - 1: 1 - n] if n >= 2 else ParseList
                    for pos, words in enumerate(ParseList):
                        tmp_words += words + u'  '
                else:
                    tmp_words += sent + u'  '
            test_result_file.write(tmp_words + '\n')
            tmp_words = u''

        test_file.close()
        test_result_file.close()
        # print('sent: {}\tchar: {}\t'.format(sent_count, char_count), end="")
        speed = int(char_count / (time.perf_counter() - start_t) / 1000)
        if log:
            print(f'sp={speed}k/s\t', end="")
        return speed

    # 使用1-gram计算给定切分序列的概率
    def UniSegProb(self, ParseList, delta=1):
        p = 0
        # 1元切分概率：由于概率很小，对概率连乘做了取对数处理，转化为加法
        for pos, word in enumerate(ParseList):
            # 平滑策略: +δ平滑
            # 平滑1阶未登录词 （uni-gram情况下只有1阶未登录词）
            numerator = 1.0 * delta
            if word in self._WordDict:
                numerator += self._WordDict[word]
            denominator = self._UniWordCount + self._UniVocabSize * delta
            p += math.log(numerator / denominator)
        return p

    # 使用2-gram计算给定切分序列的概率
    def BiSegProb(self, ParseList, delta=1):
        p = 0
        # 2元切分概率：由于概率很小，对条件概率连乘做了取对数处理，转化为加法
        for pos, word in enumerate(ParseList[:-1]):
            word1, word2 = word, ParseList[pos + 1]
            # 平滑策略: +δ平滑
            # TODO LIST: +δ平滑 => 退化的1-gram词频平滑（绝对减值法AD）
            if word1 not in self._NextWord:
                # 平滑2阶未登录词
                p += math.log(1.0 / self._BiVocabSize)
            else:
                # 平滑1阶未登录词
                numerator = 1.0 * delta
                if word2 in self._NextWord[word1]:
                    numerator += self._NextWord[word1][word2]
                denominator = self._BiWordCount[word1] + self._BiVocabSize * delta
                p += math.log(numerator / denominator)
        return p

    # 使用3-gram计算给定切分序列的概率
    def TriSegProb(self, ParseList, delta=1):
        p = 0
        # 3元切分概率：由于概率很小，对条件概率连乘做了取对数处理，转化为加法
        for pos, word in enumerate(ParseList[:-2]):
            word1, word2, word3 = word, ParseList[pos + 1], ParseList[pos + 2]
            # 平滑策略: +δ平滑
            # TODO LIST: +δ平滑 => 退化的1-gram词频平滑（绝对减值法AD）
            if word1 not in self._NextNextWord or word2 not in self._NextNextWord[word1]:
                # 平滑2阶/3阶未登录词
                p += math.log(1.0 / self._TriVocabSize)
            else:
                # 平滑1阶未登录词
                numerator = 1.0 * delta
                if word3 in self._NextNextWord[word1][word2]:
                    numerator += self._NextNextWord[word1][word2][word3]
                denominator = self._TriWordCount[word1][word2] + self._TriVocabSize * delta
                p += math.log(numerator / denominator)
        return p

    # 正向最大匹配
    def PreMax(self, sentence):
        cur, tail = 0, span
        ParseList = []
        while cur < tail and cur <= len(sentence):
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

    # 逆向最大匹配
    def PostMax(self, sentence):
        cur = len(sentence) - span
        tail = len(sentence)
        if cur < 0:
            cur = 0

        ParseList = []
        while cur < tail and tail > 0:
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
def train_and_test(dataset="pku", n=1, delta=1, outer_rules=True, method="cross", inner_rules=True, log=True):
    # data files
    train_file = f"{train_dir}/{dataset}_training.utf8"
    test_file = f"{test_dir}/{dataset}_test.utf8"
    pred_file = f"{pred_dir}/{dataset}_test_pred_{model_type}.utf8"

    # train
    p = N_Gram(train_file, test_file, pred_file)
    p.Training()
    # test
    if log:
        print(f"n={n}\tδ={delta}\touterR={outer_rules}\tmethod={method}\tinnerR={inner_rules}\t", end="")
    p.Segmentation(n, delta, outer_rules, method, inner_rules, log)
    # eval
    eval(model_type, dataset, log)


if __name__ == '__main__':
    train_and_test("pku")
    train_and_test("msr")
