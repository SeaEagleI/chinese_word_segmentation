import torch
import torch.nn as nn
# from torchcrf import CRF


class BiLSTM_Softmax(nn.Module):

    def __init__(self, embedding_size, hidden_size, vocab_size, num_labels, num_layers, lstm_drop_out, nn_drop_out,
                 pretrained_embedding=False, embedding_weight=None):
        super(BiLSTM_Softmax, self).__init__()
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.nn_drop_out = nn_drop_out
        # nn.Embedding: parameter size (num_words, embedding_dim)
        # for every word id, output a embedding for this word
        # input size: N x W, N is batch size, W is max sentence len
        # output size: (N, W, embedding_dim), embedding all the words
        if pretrained_embedding:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight)
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(
            input_size=embedding_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=lstm_drop_out if num_layers > 1 else 0,
            bidirectional=True
        )
        if nn_drop_out > 0:
            self.dropout = nn.Dropout(nn_drop_out)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, unigrams, bigrams, loss_mask, labels=None, training=True):
        uni_embeddings = self.embedding(unigrams)
        bi_embeddings = self.embedding(bigrams)
        outputs = torch.cat([uni_embeddings, bi_embeddings], dim=-1)
        sequence_output, _ = self.bilstm(outputs)
        if training and self.nn_drop_out > 0:
            sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,)
        if labels is not None:
            # loss_mask = labels.gt(-1)
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if loss_mask is not None:
                # 只留下label存在的位置计算loss
                active_loss = loss_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs
