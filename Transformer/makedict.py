import utils
import pickle
import os
import argparse
from utils import is_dataset_tag, make_sure_path_exists

parser = argparse.ArgumentParser()
# --datasets是建立词典需要的数据集
parser.add_argument('--datasets', type=str, default='joint-sighan2005', help='two methods: all sighan-2005 or just pku/msr')
# parser.add_argument('--out_path', type=str, default='dict.pkl', help='output directory name')
args = parser.parse_args()

datasets = args.datasets
out_path = datasets + ".pkl"

path=f"data/{datasets}/raw/train-all.txt"

# 词典的词全部来自于train-all, dev和train的词也在里面
# 所以训练期间oov score会一直为0，因为分母为0，等到测试阶段才能计算oov

dic={}
tokens={}
with open(path, "r", encoding="utf-16") as f:
    for line in f.readlines():
        cur=line.strip().split(" ")
        name=cur[0][1:-1]
        if dic.get(name) is None:
            dic[name]=set()
            tokens[name]=0
        tokens[name]+=len(cur[1:-1])
        dic[name].update(cur[1:-1])

for i in list(dic.keys()):
    print(i,len(dic[i]),tokens[i])
with open(out_path,"wb") as outfile:
    pickle.dump(dic,outfile)
