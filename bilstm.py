import logging

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from utils.DataAgent import DataAgent
from utils.Encoder import WordEmbeddingEncoder

logging.basicConfig(format='%(asctime)s - %(levelname)s[%(name)s] - %(message)s')
logger = logging.getLogger('Main')
logger.setLevel(logging.DEBUG)

da = DataAgent()
we = WordEmbeddingEncoder(da, mode='lstm')


class BiLSTM(torch.nn.Module):
    def __init__(self, we: WordEmbeddingEncoder, hidden_size=64, out_size=3, dropout=0.1):
        super(BiLSTM, self).__init__()
        self.logger = logging.getLogger('BiLSTM')
        vocab_size = len(we.alphabet)
        emb_size = we.zero_vec.size
        emb_mat = we.make_emb_layer()
        self.embedding = torch.nn.Embedding(vocab_size, emb_size, padding_idx=0)
        if emb_mat is not None:
            self.logger.info('Initializing embedding layer with pretrained vectors.')
            self.embedding.weight.data.copy_(torch.from_numpy(emb_mat))
            self.embedding.weight.requires_grad = False
        else:
            self.logger.warning('Embedding matrix is None, continue with random embedding layer.')
        self.lstm = torch.nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = torch.nn.Linear(2 * hidden_size, out_size)

    def forward(self, x: PackedSequence):
        x, lengths = pad_packed_sequence(x, batch_first=True)
        x = pack_padded_sequence(self.embedding(x), lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x = pad_packed_sequence(x, batch_first=True)[0]
        x = pack_padded_sequence(self.fc(x), lengths, batch_first=True, enforce_sorted=False)
        return x


net = BiLSTM(we)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr=0.01)

training_data = da.range_uid('19980101', '19980120')
validation_data = da.range_uid('19980121', '19980125')
test_data = da.range_uid('19980126', '19980131')

epochs = 100
batch_size = 2048

for e in range(epochs):
    net.train()
    pred_list = []
    label_list = []
    for b in range(int(len(training_data) / batch_size) + 1):
        packed_vec, packed_labels = we.batch_to_packed_idx(training_data[batch_size * b:batch_size * (b + 1)])
        mask = (packed_vec == we.vec_dict)
        label_list.extend([label.get_bio_label() for label in packed_labels])
        pred_list, lengths = pad_packed_sequence(net(packed_vec), batch_first=True)[0]
        pred_list.extend(b)
    pred_list = torch.cat(pred_list, 0)
    # todo train
    pass

# t, _ = we.batch_to_packed_idx(['19980101-01-003-001', '19980101-01-003-002'])
