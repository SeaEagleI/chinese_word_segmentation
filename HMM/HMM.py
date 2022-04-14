# coding:utf-8
import numpy as np
from config import train_dir, test_dir, pred_dir

# model params
model_type = "hmm"


# learn HMM params from train set (supervised paradigm)
def train(fileName):
    # HMM模型由三要素决定 lambda=（A，B，pi）
    # A为状态转移矩阵
    # B为观测概率矩阵
    # pi为初始状态概率向量

    # 在该函数中，我们需要通过给定的训练数据（包含S个长度相同的观测序列【每一句话】和对应的状态序列【每一句话中每个词的词性】

    # 在中文分词中，包含一下集中状态（词性）
    # B：词语的开头（单词的头一个字）
    # M：中间词（即在一个词语的开头和结尾之中）
    # E：单词的结尾（即单词的最后一个字）
    # S：单个字

    # 定义一个状态映射字典。方便我们定位状态在列表中对应位置
    status2num = {'B': 0, 'M': 1, 'E': 2, 'S': 3}

    # 定义状态转移矩阵。总共4个状态，所以4x4
    A = np.zeros((4, 4))

    # 定义观测概率矩阵
    # 在ord为所有unicode字符进行编码[example: ord('我')=25105]，大小为65536，总共4个状态
    # 所以B矩阵4x65536
    # 就代表每一种状态（词性）得到观测状态（字）
    B = np.zeros((4, 65536))

    # 初始状态，每一个句子的开头只有4中状态（词性）
    PI = np.zeros(4)

    with open(fileName, encoding='utf-8') as file:
        # 每一行读取
        # 如某一行语料为：   迈向  充满  希望  的  新  世纪 。
        # 语料库为我们进行好了切分，每一个词语用空格隔开。
        # 那么在这其中，我们将每个词语切分（包括标点符号）放在列表中。
        # 然后遍历列表每一个元素
        # 当列表词语长度为1的时候，如 '的'字，那么我们就认为状态为S（单个字）
        # 当列表长度为2的时候，如'迈向'，我们认为'迈'为B，'向'为E
        # 当长度为3以上时候，如'实事求是'，我们认为'实'为B，'事求'两个字均为M，'是'为E

        # 我们遍历完毕所有的语料，就可以按照公式10.39,40,41来获取A，B，PI
        # 其实这三个公式的本质是统计出频数/总数
        # 如10.39，公式上半部分是从1-T-1时刻，t时刻状态为qi，t+1时刻为qj状态的总概率。
        # 那么由似然可以知道，该总概率是由 1-T-1时刻，t时刻状态为qi，t+1时刻为qj状态出现次数/ 观测序列对应状态序列总数
        # 下方也类似。 两者相除，分母均为观测序列对应状态序列总数，可以相互抵消
        # 就可以变为 1-T-1时刻，t时刻状态为qi，t+1时刻为qj状态出现次数/1-T-1时刻，t时刻状态为qi出现次数
        # 所以一下我们只需要统计出现频数，然后除总次数即可。

        # date: 12.26
        # 刚刚上面的说错了QAQ 今天在回顾的时候，发现那时候自己理解还不够啊
        # 我们的训练集是已经标注完毕了，所以我们的学习算法是有监督学习
        # 所以就是直接极大似然，频数/总数   就可以得出模型三要素的参数了
        for line in file.readlines():
            wordStatus = []  # 用于保存该行所有单词的状态
            words = line.strip().split()  # 除去前后空格，然后依照中间空格切分为单词

            for i, word in enumerate(words):

                # 根据长度判断状态
                if len(word) == 1:
                    status = 'S'  # 保存每一个单词状态
                    # 使用ord找到该字对应编码
                    # 更新B矩阵
                    # B代表了每一个状态到对应观测结果的可能性
                    # 先统计频数
                    code = ord(word)
                    B[status2num[status[0]]][code] += 1

                else:
                    # 当长度为2，M*0。这样可以一起更新
                    status = 'B' + (len(word) - 2) * 'M' + 'E'
                    # 使用ord找到该字对应编码
                    # 更新B矩阵
                    # B代表了每一个状态到对应观测结果的可能性
                    # 先统计频数
                    for s in range(len(word)):
                        code = ord(word[s])
                        B[status2num[status[s]]][code] += 1

                # i==0意味着这是句首。我们需要更新PI中每种状态出现次数
                if i == 0:
                    # status[0]表示这行第一个状态
                    # status2num将其映射到list对应位置
                    PI[status2num[status[0]]] += 1

                # 使用extend，将status中每一个元素家在列表之中。而不是append直接将整个status放在后面
                wordStatus.extend(status)

            # 遍历完了一行，然后更新矩阵A
            # A代表的是前一个状态到后一个状态的概率
            # 我们先统计频数
            for i in range(1, len(wordStatus)):
                # wordStatus获得状态，使用status2num来映射到正确位置
                A[status2num[wordStatus[i - 1]]][status2num[wordStatus[i]]] += 1

    # 读取完毕文件，频数统计完成
    # 接下来计算概率
    # 我们面临的问题是：
    # 1.如果句子较长，许多个较小的数值连乘，容易造成下溢。对于这种情况，我们常常使用log函数解决。
    # 但是，如果有一些没有出现的词语，导致矩阵对应位置0，那么测试的时候遇到了，连乘中有一个为0，整体就为0。
    # 但是log0是不存在的，所以我们需要给每一个0的位置加上一个极小值（-3.14e+100)，使得其有定义。

    # 计算PI向量
    total = sum(PI)
    for i in range(len(PI)):
        if PI[i] == 0:
            PI[i] = -3.14e+100
        else:
            # 别忘了去取对数
            PI[i] = np.log(PI[i] / total)

    # 计算A矩阵
    # 要注意每一行的和为1，即从某个状态向另外4个状态转移概率只和为1
    # 最后我们取对数
    for i in range(len(A)):
        total = sum(A[i])
        for j in range(len(A[i])):
            if A[i][j] == 0:
                A[i][j] = -3.14e+100
            else:
                A[i][j] = np.log(A[i][j] / total)
    # 更新B矩阵
    # B矩阵中，每一行只和为1
    # 即某一个状态到所有观测结果只和为1
    # 最后我们取对数
    for i in range(len(B)):
        total = sum(B[i])
        for j in range(len(B[i])):
            if B[i][j] == 0:
                B[i][j] = -3.14e+100
            else:
                B[i][j] = np.log(B[i][j] / total)

    # 返回三个参数
    return PI, A, B


# run word segmentation based on learned HMM params
def word_partition(HMM_parameter, article):
    '''
    使用维特比算法进行预测（即得到路径中每一个最有可能的状态）
    :param HMM_parameter: PI,A,B隐马尔可夫模型三要素
    :param article: 需要分词的文章,以数组的形式传入，每一个元素是一行
    :return: 分词后的文章
    '''

    PI, A, B = HMM_parameter
    article_partition = []  # 分词之后的文章

    # 我们需要计算的是Ψ（psi），δ（delta）
    # delta对应于公式10.44,45.psi对应于公式10.46

    for line in article:
        # 定义delta，psi
        # delta一共长度为每一行长度，每一位有4种状态
        delta = [[0 for _ in range(4)] for _ in range(len(line))]
        # psi同理
        psi = [[0 for _ in range(4)] for _ in range(len(line))]

        for t in range(len(line)):
            if t == 0:
                # 初始化psi
                psi[t][:] = [0, 0, 0, 0]
                for i in range(4):
                    # !!! 注意这里是加号，因为之前log处理了
                    delta[t][i] = PI[i] + B[i][ord(line[t])]

            # 依照两个公式更细delta和psi
            # 注意每一个时刻的delta[t][i]代表的是到当前时刻t，结束状态为i的最有可能的概率
            # psi[t][i]代表的是当前时刻t，结束状态为i，在t-1时刻最有可能的状态（S，M，E，B）
            else:
                for i in range(4):
                    # 一共4中状态，就不写for循环一个个求出在的max了，直接写成列表了

                    # !!! 划重点，注意这里概率之间的计算用的加号
                    # 因为之前我们进行了log处理，所以之前的概率相乘变成了log相加

                    # temp=[delta[t-1][0]+A[0][i],delta[t-1][1]+A[1][i],delta[t-1][2]+A[2][i],delta[t-1][3]+A[3][i]]
                    temp = [delta[t - 1][j] + A[j][i] for j in range(4)]  # 写成列表生成式吧，短一点。和上面一样的
                    # 求出max在乘以b
                    # b[i][ot]中，ot就是观测结果，即我们看到的字
                    # 我们使用ord将其对应到编码，然后就可以获得他在观测概率矩阵中，由状态i到观测结果（ord（line[t]))的概率了
                    delta[t][i] = max(temp) + B[i][ord(line[t])]

                    # 求psi
                    # 可以注意到，psi公式中，所求的是上一个最有可能的概率
                    # argmax中的值就是上方的temp，所以我们只需要获得temp最大元素的索引即可
                    psi[t][i] = temp.index(max(temp))

        # 遍历完毕这一行了，我们可以计算每个词对应的状态了
        # 依照维比特算法步骤4，计算最优回溯路径
        # 我们保存的是索引，0，1，2，3。对应与B，M，E，S
        status = []  # 用于保存最优状态链

        # 非空检查
        if line != '':
            # 计算最优状态链
            # 最优的最后一个状态
            It = delta[-1].index(max(delta[-1]))
            status.append(It)
            # 这是后向的计算该最优路径
            # 所以我们使用insert，在列表最前方插入当前算出的最优节点。
            for t in range(len(delta) - 2, -1, -1):
                # status[0]保存的是所求的当前t时刻的后一时刻（t+1），最有可能的状态
                # psi[t][i]表示t时刻，状态为i，t-1时刻最有可能的状态
                # 所以用psi[t+1][status[0]]就可以得出t时刻最有可能的状态
                It = psi[t + 1][status[0]]
                status.insert(0, It)

        # 计算出了所有所有时刻最有可能的状态之后，进行分词
        # 遇到S，E我们就要在该词之后输出｜
        # 例如 我今天很开心 对应 S，B，E，S，B，E 输出 我｜今天｜很｜开心｜。
        # 只需要注意这一行最后不输出｜即可
        line_partition = ''
        for t in range(len(line)):
            line_partition += line[t]
            if (status[t] == 2 or status[t] == 3) and t != len(line) - 1:
                line_partition += '  '
        # 结束输出，换行
        article_partition.append(line_partition)

    return article_partition


# load train & test set
def loadArticle(fileName):
    '''
    读取测试文章
    :param fileName: 文件名
    :return: 处理之后的文章
    '''
    # 我们需要将其空格去掉
    with open(fileName, encoding='utf-8') as file:
        # 按行读取
        test_article = []
        for line in file.readlines():
            # 去除空格，以及换行符
            line = line.strip()
            test_article.append(line)
    return test_article


# train & test loop
def train_and_test(dataset="pku"):
    # data_files
    train_file = f"{train_dir}/{dataset}_training.utf8"
    test_file = f"{test_dir}/{dataset}_test.utf8"
    pred_file = f"{pred_dir}/{dataset}_test_pred_{model_type}.utf8"

    # training
    param = train(train_file)
    # testing
    article = loadArticle(test_file)
    print(f"dataset: {dataset} | test_size: {len(article)}")
    article_partition = word_partition(param, article)
    # print(article_partition)
    open(pred_file, "w+", encoding="utf-8")\
        .write("\n".join(article_partition) + "\n")

    # **********自定义测试***************
    # 请输出测试语句行数2
    # 请输入语句：今天很开心
    # 请输入语句：写完了隐马尔可夫模型中文分词
    # ['今天|很|开心', '写|完|了|隐马尔可夫|模型|中文|分词']
    # print('**********自定义测试***************')
    # line_num = int(input('请输出测试语句行数'))
    # article_cumstmize = []
    # for i in range(line_num):
    #     sentence = input('请输入语句：')
    #     article_cumstmize.append(sentence)
    # article_cumstmize_partition = word_partition(param, article_cumstmize)
    # print(article_cumstmize_partition)


if __name__ == '__main__':

    train_and_test("pku")
    train_and_test("msr")
