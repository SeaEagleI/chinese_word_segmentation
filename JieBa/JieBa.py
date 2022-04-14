from config import test_dir, pred_dir
from evaluate import eval
import jieba

# model params
model_type = "jieba"


# test loop
def test(dataset="pku"):
    test_file = f"{test_dir}/{dataset}_test.utf8"
    pred_file = f"{pred_dir}/{dataset}_test_pred_{model_type}.utf8"
    gold_file = f"{test_dir}/{dataset}_test_gold.utf8"

    # test
    with open(pred_file, "w+", encoding="utf-8") as f:
        for line in open(test_file, encoding="utf-8").readlines():
            line = line.strip()
            f.write("  ".join(jieba.cut(line)) + "\n")

    # eval
    eval(model_type, dataset)


if __name__ == '__main__':

    test("pku")
    test("msr")
