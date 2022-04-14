from config import test_dir, pred_dir


# self-implemented Evaluate Script
def eval(model_type, dataset):
    pred_file = f"{pred_dir}/{dataset}_test_pred_{model_type}.utf8"
    gold_file = f"{test_dir}/{dataset}_test_gold.utf8"
    pred_f = open(pred_file, encoding="utf-8")
    gold_f = open(gold_file, encoding="utf-8")

    right_cnt = 0.0   # TP
    result_cnt = 0.0  # TP+FP
    gold_cnt = 0.0    # TP+FN
    for line1, line2 in zip(pred_f, gold_f):
        result_list = [w for w in line1.strip().split(' ') if w]
        gold_list = [w for w in line2.strip().split(' ') if w]

        result_cnt += len(result_list)
        gold_cnt += len(gold_list)
        for words in result_list:
            if words in gold_list:
                right_cnt += 1.0
                gold_list.remove(words)

    # 分词任务的TN样例隐含, 因为能在分词结果中看到的词都是TP/FP/FN，看不到的才是TN（看不到的所有可能分词数存在组合爆炸）
    p = right_cnt / result_cnt  # TP/(TP+FP)
    r = right_cnt / gold_cnt    # TP/(TP+FN)
    F = 2.0 * p * r / (p + r)

    print(f'\n{model_type} - {dataset}:')
    print('right_cnt: \t\t', right_cnt)
    print('result_cnt: \t', result_cnt)
    print('gold_cnt: \t\t', gold_cnt)
    print('P: \t\t', p)
    print('R: \t\t', r)
    print('F: \t\t', F)


if __name__ == "__main__":
    datasets = ["pku", "msr"]
    model_types = ["hmm", "ngram"]
    for model in model_types:
        for dataset in datasets:
            eval(model, dataset)
