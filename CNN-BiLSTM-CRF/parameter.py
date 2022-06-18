hidden_size = 128   # LSTM cell 深度为128
vacab_size = 5000
embedding_size = 100
units = 4

filters_num = 128   # 卷积核数目
half_kernel_size = 4     # 卷积核大小

keep_pro = 0.5
learning_rate = 0.006
lr = 0.6
clip = 5.0

epochs = 8
batch_size = 64

# ！！！！将文本中所有句子长度统一设置为600字符！！！！！
seq_max_len = 600
# train = './data/WordSeg.txt'
# train = './data/pku_training.utf8'
train = './data/msr_training.utf8'
test = './data/test.txt'
# eva = './data/eva.txt'
eva = './data/msr_test_'