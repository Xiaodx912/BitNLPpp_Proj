import logging
import time
import json

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from utils.DataAgent import DataAgent
from utils.Encoder import WordEmbeddingEncoder
from utils.util_func import calc_masked_result, make_ans_eval_mat, calc_tri_classification_f1

logging.basicConfig(format='%(asctime)s - %(levelname)s[%(name)s] - %(message)s')
logger = logging.getLogger('Main')
logger.setLevel(logging.DEBUG)

da = DataAgent()
we = WordEmbeddingEncoder(da, mode='lstm')


class BiLSTM(torch.nn.Module):
    def __init__(self, enc: WordEmbeddingEncoder, hidden_size=64, out_size=3, dropout=0):
        super(BiLSTM, self).__init__()
        self.logger = logging.getLogger('BiLSTM')
        vocab_size = len(enc.alphabet)
        emb_size = enc.zero_vec.size
        emb_mat = enc.make_emb_layer()
        self.embedding = torch.nn.Embedding(vocab_size, emb_size, padding_idx=0)
        if emb_mat is not None:
            self.logger.info('Initializing embedding layer with pretrained vectors.')
            self.embedding.weight.data.copy_(torch.from_numpy(emb_mat))
            # self.embedding.weight.requires_grad = False
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = BiLSTM(we).to(device)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr=0.05)

training_data = da.range_uid('19980101', '19980120')
validation_data = da.range_uid('19980121', '19980125')
test_data = da.range_uid('19980126', '19980131')

epochs = 100
batch_size = 2048

train_data_size = 0
for uid in training_data:
    train_data_size += len(da.main_data[uid].split())

train_log = []

for e in range(epochs):
    net.train()
    # pred_list = []
    # label_list = []
    optim.zero_grad()
    loss_sum = 0
    log = {'e': e, 'LR': float(optim.state_dict()['param_groups'][0]['lr'])}
    for batch_n in range(int(len(training_data) / batch_size) + 1):
        packed_vec, bio_labels = we.batch_to_packed_idx(training_data[batch_size * batch_n:batch_size * (batch_n + 1)])
        pred, label = calc_masked_result(packed_vec.to(device), bio_labels, net, device)
        loss = criterion(pred, label)
        loss_sum += (loss * (len(pred) / train_data_size)).to('cpu')
        print(f'e:{e}\tb:{batch_n}\tl:{loss}')
        log[f'b{batch_n}'] = {'loss': float(loss)}
        loss.backward()
        del label, pred, packed_vec, bio_labels, loss
        torch.cuda.empty_cache()
    optim.step()
    # todo train
    if True:
        net.eval()
        pred_list = []
        label_list = []
        for batch_n in range(int(len(validation_data) / batch_size) + 1):
            packed_vec, bio_labels = we.batch_to_packed_idx(
                validation_data[batch_size * batch_n:batch_size * (batch_n + 1)])
            pred, label = calc_masked_result(packed_vec.to(device), bio_labels, net, device)
            _, pred = torch.max(pred, dim=1)
            pred_list.append(pred.to('cpu'))
            label_list.append(label.to('cpu'))
            del packed_vec, bio_labels, pred, label, _
            torch.cuda.empty_cache()
        ans_mat = make_ans_eval_mat(torch.cat(pred_list), torch.cat(label_list).to('cpu'))
        print(ans_mat)
        f1 = calc_tri_classification_f1(ans_mat)
        log.update({'test_result': ans_mat.tolist(), 'loss': float(loss_sum)})
        log.update(f1)
        print(loss_sum, f1)
        train_log.append(log)
        # todo eval
        if (e + 1) % 20 == 0:
            torch.save(net.state_dict(), 'model/e{}_L{:.4f}_BOF{:.4f}_{}.pkl'
                       .format(e, loss_sum, f1['BO'], int(time.time())))

f = open(f'model/log_{int(time.time())}.json', 'w')
f.write(json.dumps(train_log))
f.close()

