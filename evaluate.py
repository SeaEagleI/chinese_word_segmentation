from config import test_dir, pred_dir


# self-implemented Evaluate Script
def eval(model_type, dataset, log=False):
    pred_file = f"{pred_dir}/{dataset}_test_pred_{model_type}.utf8"
    gold_file = f"{test_dir}/{dataset}_test_gold.utf8"
    pred_f = open(pred_file, encoding="utf-8")
    gold_f = open(gold_file, encoding="utf-8")

    right_cnt = 0.0  # TP
    result_cnt = 0.0  # TP+FP
    gold_cnt = 0.0  # TP+FN
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
    P = right_cnt / result_cnt * 100  # TP/(TP+FP)
    R = right_cnt / gold_cnt * 100    # TP/(TP+FN)
    F = 2.0 * P * R / (P + R)

    if log:
        # print(f"right: {right_cnt}\tresult: {result_cnt}\tgold: {gold_cnt}")
        print('{}\t{}\tP={:.2f}\tR={:.2f}\tF={:.2f}'.format(model_type, dataset, P, R, F))
    return P, R, F


if __name__ == "__main__":
    datasets = ["pku", "msr"]
    model_types = ["hmm", "ngram", "jieba"]
    for model in model_types:
        for dataset in datasets:
            eval(model, dataset)
