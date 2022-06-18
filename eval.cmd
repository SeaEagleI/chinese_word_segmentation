@echo off

for %%M in (ngram) do ^
for %%D in (pku) do ^
perl score.pl data/train/%%D_training_words.utf8 data/test/%%D_test_gold.utf8 ^
data/preds/%%D_test_pred_%%M.utf8 > data/scores/%%D_score_%%M.utf8 ^
&& echo eval ended for %%M_%%D

perl score.pl data/train/pku_training_words.utf8 data/test/pku_test_gold.utf8 data/preds/pku_test_pred_ngram.utf8 > data/scores/pku_score_ngram.utf8

perl score.pl data/train/pku_training_words.utf8 data/test/pku_test_gold.utf8 data/unigram_pku_seg.txt > data/pku_score_hcc.utf8