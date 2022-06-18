import torch
import logging
from torch.utils.data import Dataset


class SegDataset(Dataset):
    def __init__(self, words, labels, vocab, label2id):
        self.vocab = vocab
        self.dataset = self.preprocess(words, labels)
        self.label2id = label2id

    def preprocess(self, words, labels):
        """convert the data to ids"""
        processed = []
        for (word, label) in zip(words, labels):
            word_id = [self.vocab.word_id(u_) for u_ in word]
            label_id = [self.vocab.label_id(l_) for l_ in label]
            processed.append((word_id, label_id))
        logging.info("-------- Process Done! --------")
        return processed

    def __getitem__(self, idx):
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        return len(self.dataset)

    def get_long_tensor(self, words, labels, batch_size):
        token_len = max([len(x) for x in labels])
        word_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        label_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        mask_tokens = torch.ByteTensor(batch_size, token_len).fill_(0)

        for i, s in enumerate(zip(words, labels)):
            word_tokens[i, :len(s[0])] = torch.LongTensor(s[0])
            label_tokens[i, :len(s[1])] = torch.LongTensor(s[1])
            mask_tokens[i, :len(s[0])] = torch.tensor([1] * len(s[0]), dtype=torch.uint8)

        return word_tokens, label_tokens, mask_tokens

    def collate_fn(self, batch):

        words = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        lens = [len(x) for x in labels]
        batch_size = len(batch)

        word_ids, label_ids, input_mask = self.get_long_tensor(words, labels, batch_size)

        return [word_ids, label_ids, input_mask, lens]
