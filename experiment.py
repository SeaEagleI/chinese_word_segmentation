import numpy as np
from tqdm import tqdm
from config import train_dir, test_dir, pred_dir
from evaluate import eval
import HMM.HMM as hmm
import N_Gram.Ngram as ngram

# params
eps = 1e-5
model_list = ["hmm", "ngram"]
gram_list = [1]
delta_list = np.arange(0, 1.1, 0.1)
dataset_list = ["pku", "msr"]


# grid_search for hmm & ngram
def grid_search(model, dataset):
    assert model in model_list
    if model == "hmm":
        scores = {}
        for delta in tqdm(delta_list, desc=f"{model} {dataset}"):
            scores[delta] = hmm.train_and_test(dataset, delta)[-1]
        best_delta, best_F = max(list(scores.items()), key=lambda x: x[1])
        print(f"\n{model}\t{dataset}\tδ={best_delta}\tF={best_F:.2f}")
    else:
        # # data files
        # train_file = f"{train_dir}/{dataset}_training.utf8"
        # test_file = f"{test_dir}/{dataset}_test.utf8"
        # pred_file = f"{pred_dir}/{dataset}_test_pred_{model}.utf8"
        # # train
        # p = ngram.N_Gram(train_file, test_file, pred_file)
        # p.Training()
        # # test: delta
        # delta_list[0] = eps
        # for n in gram_list:
        #     scores = {}
        #     for delta in tqdm(delta_list, desc=f"{n}-gram {dataset}"):
        #         p.Segmentation(n, delta)
        #         scores[delta] = eval(model, dataset)[-1]
        #     best_delta, best_F = max(list(scores.items()), key=lambda x: x[1])
        #     print(f"\n{model}\t{dataset}\tn={n}\tδ={best_delta}\tF={best_F:.2f}")
        # test: delta
        delta_list[0] = eps
        n, delta, bools = 1, -1, [True, False]
        for outR in bools:
            for method in ["hybrid", "all-cut"]:
                for inR in bools:
                    ngram.train_and_test(dataset, n=n, delta=delta, outer_rules=outR, method=method, inner_rules=inR)


# parameter experiment
# 无规则情况下的hmm实验
# grid_search("hmm", "pku")
# grid_search("hmm", "msr")

# 无规则情况下的n-gram实验
# grid_search("ngram", "pku")
# grid_search("ngram", "msr")

ngram.train_and_test("pku", n=1, delta=1e-5, outer_rules=False, method="prepost", inner_rules=False)
ngram.train_and_test("msr", n=1, delta=0.4, outer_rules=False, method="prepost", inner_rules=False)
ngram.train_and_test("pku", n=2, delta=0.1, outer_rules=False, method="prepost", inner_rules=False)
ngram.train_and_test("msr", n=2, delta=0.1, outer_rules=False, method="prepost", inner_rules=False)
ngram.train_and_test("pku", n=3, delta=0.2, outer_rules=False, method="prepost", inner_rules=False)
ngram.train_and_test("msr", n=3, delta=0.1, outer_rules=False, method="prepost", inner_rules=False)

# 有规则情况下的1-gram混合和全切分实验
grid_search("ngram", "pku")
grid_search("ngram", "msr")

# ngram.train_and_test("pku", n=1, delta=-1, outer_rules=True, method="hybrid", inner_rules=True)
# ngram.train_and_test("pku", n=1, delta=-1, outer_rules=True, method="all-cut", inner_rules=True)
# ngram.train_and_test("pku", n=1, delta=-1, outer_rules=True, method="all-cut", inner_rules=False)
# ngram.train_and_test("pku", n=1, delta=-1, outer_rules=False, method="all-cut", inner_rules=True)
# ngram.train_and_test("pku", n=1, delta=-1, outer_rules=False, method="all-cut", inner_rules=False)


