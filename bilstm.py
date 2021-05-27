import logging

import numpy as np
import torch

from utils.DataAgent import DataAgent
from utils.Encoder import WordEmbeddingEncoder

x = [[1, 2, 0, 0], [9, 0, 0, 0], [3, 3, 4, 0], [5, 6, 7, 8]]
lengths = [2, 1, 3, 4]
x = torch.autograd.Variable(torch.Tensor(x))
lengths = torch.Tensor(lengths).int()
_, idx_sort = torch.sort(lengths, dim=0, descending=True)
_, idx_unsort = torch.sort(idx_sort, dim=0)

sorted_x = x.index_select(0, idx_sort)
sorted_lengths = lengths.index_select(0, idx_sort)
x_packed = torch.nn.utils.rnn.pack_padded_sequence(input=sorted_x, lengths=sorted_lengths, batch_first=True)

logging.basicConfig(format='%(asctime)s - %(levelname)s[%(name)s] - %(message)s')
logger = logging.getLogger('Main')
logger.setLevel(logging.DEBUG)

da = DataAgent()
we = WordEmbeddingEncoder(da)

class BiLSTM(torch.nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size, dropout=0.1):
        super(BiLSTM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_size)
        self.lstm = torch.nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(2 * hidden_size, out_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, lengths):  # [b,l,emb_size ]
        emb = self.dropout(self.embedding(x))
        emb = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True)
        emb, _ = self.lstm(emb)
        emb, _ = torch.nn.utils.rnn.pad_packed_sequence(emb, batch_first=True, padding_value=0.,
                                                        total_length=x.shape[1])
        scores = self.fc(emb)

        return scores

