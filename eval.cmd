@echo off

for %%M in (hmm,ngram) do ^
for %%D in (pku,msr) do ^
perl score.pl data/train/%%D_training_words.utf8 data/test/%%D_test_gold.utf8 ^
data/preds/%%D_test_pred_%%M.utf8 > data/scores/%%D_score_%%M.utf8 ^
&& echo eval ended for %%M_%%D
