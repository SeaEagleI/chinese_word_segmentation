import numpy as np
from tqdm import tqdm
from config import train_dir, test_dir, pred_dir
from evaluate import eval
import HMM.HMM as hmm
import N_Gram.PrePostNgram as ngram

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
        # data files
        train_file = f"{train_dir}/{dataset}_training.utf8"
        test_file = f"{test_dir}/{dataset}_test.utf8"
        pred_file = f"{pred_dir}/{dataset}_test_pred_{model}.utf8"
        # train
        p = ngram.PrePostNgram(train_file, test_file, pred_file)
        p.Training()
        # test
        delta_list[0] = eps
        for n in gram_list:
            scores = {}
            for delta in tqdm(delta_list, desc=f"{n}-gram {dataset}"):
                p.Segmentation(n, delta)
                scores[delta] = eval(model, dataset)[-1]
            best_delta, best_F = max(list(scores.items()), key=lambda x: x[1])
            print(f"\n{model}\t{dataset}\tn={n}\tδ={best_delta}\tF={best_F:.2f}")


# parameter experiment
# grid_search("hmm", "pku")
# grid_search("hmm", "msr")
# grid_search("ngram", "pku")
# grid_search("ngram", "msr")
# ngram.train_and_test("pku", n=1, delta=1e-05)
# ngram.train_and_test("pku", n=2, delta=0.1)
# ngram.train_and_test("pku", n=3, delta=0.2)
# ngram.train_and_test("msr", n=1, delta=0.4)
# ngram.train_and_test("msr", n=2, delta=0.1)
# ngram.train_and_test("msr", n=3, delta=0.1)

ngram.train_and_test("pku", n=1, delta=-1, use_rules=True, all_cut=True)
ngram.train_and_test("pku", n=1, delta=-1, use_rules=False, all_cut=True)
ngram.train_and_test("msr", n=1, delta=-1, use_rules=True, all_cut=True)
ngram.train_and_test("msr", n=1, delta=-1, use_rules=False, all_cut=True)


# dataset cross/merge experiment



