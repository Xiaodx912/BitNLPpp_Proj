import json
import logging
import time

import numpy as np
import torch

from utils.DataAgent import DataAgent
from utils.Encoder import WordEmbeddingEncoder
from utils.util_func import get_range_tensor, calc_F1

logging.basicConfig(format='%(asctime)s - %(levelname)s[%(name)s] - %(message)s')
logger = logging.getLogger('Main')
logger.setLevel(logging.DEBUG)

da = DataAgent()
we = WordEmbeddingEncoder(da)

logger.info('Loading data...')
training_data = da.range_uid('19980101', '19980120')
validation_data = da.range_uid('19980121', '19980125')
test_data = da.range_uid('19980126', '19980131')

train_x_tensor, train_y_ref = get_range_tensor(we, training_data, 'bio_label')
test_x_tensor, test_y_ref = get_range_tensor(we, validation_data, 'bio_label')


class SoftMax(torch.nn.Module):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.l1 = torch.nn.Linear(150, 128)
        self.l2 = torch.nn.Linear(128, 3)

    def forward(self, x):
        x = x.view(-1, 150)
        x = torch.relu(self.l1(x))
        return self.l2(x)


train_x_tensor, train_y_ref = train_x_tensor.cuda(), train_y_ref.cuda()
test_x_tensor, test_y_ref = test_x_tensor.cuda(), test_y_ref.cuda()


def train(path=None, e_start=0, e_size=10000, learning_rate=0.05):
    net = SoftMax().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler. \
        ReduceLROnPlateau(optim, mode='min', verbose=True, patience=500, threshold_mode='rel',
                          threshold=0.001, factor=0.8, cooldown=800, min_lr=0.001)

    if path:
        logger.info(f'Loading parameter from {path}')
        net.load_state_dict(torch.load(path))

    train_log = []
    logger.info('Start training network.')

    for e in range(e_start, e_start + e_size):
        net.train()
        y_pred = net(train_x_tensor)
        loss = criterion(y_pred, train_y_ref)
        scheduler.step(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()
        log = {'epoch': e, 'loss': float(loss.item()), 'LR': float(optim.state_dict()['param_groups'][0]['lr'])}
        if (e + 1) % 10 == 0:
            net.eval()
            test_out = net(test_x_tensor)
            _, test_out_label = torch.max(test_out, dim=1)
            assert test_y_ref.size() == test_out_label.size()
            ans_mat = np.zeros((3, 3), dtype=np.int64)
            for ref_label in [0, 1, 2]:
                for out_label in [0, 1, 2]:
                    ans_mat[ref_label][out_label] = ((test_y_ref == ref_label) & (test_out_label == out_label)).sum()
            Macro_average_F1_list = np.empty(3, dtype=np.float)
            group_count = np.zeros(3, dtype=np.int64)
            for label in [0, 1, 2]:
                TP = ans_mat[label][label]
                FN = ans_mat[label].sum() - TP
                FP = ans_mat.swapaxes(0, 1)[label].sum() - TP
                group_count += np.array([TP, FN, FP], dtype=np.int64)
                Macro_average_F1_list[label] = calc_F1(TP, FN, FP)
            TP, FN, FP = group_count.tolist()
            Micro_average_F1 = calc_F1(TP, FN, FP)
            tmp = np.vstack([ans_mat[0], ans_mat[1] + ans_mat[2]]).swapaxes(0, 1)
            bo_matrix = np.vstack([tmp[0], tmp[1] + tmp[2]]).swapaxes(0, 1)
            BO_F1 = calc_F1(bo_matrix[1][1], bo_matrix[1][0], bo_matrix[0][1])
            logger.info("Epoch:{},Loss:{:.4f},micF1:{:.4f},macF1:{:.4f},BO_F1:{:.4f},LR:{:.4f}"
                        .format(e + 1, loss.item(), Micro_average_F1, Macro_average_F1_list.mean(), BO_F1,
                                optim.state_dict()['param_groups'][0]['lr']))
            log.update({'Macro_average_F1': Macro_average_F1_list.mean(), 'Micro_average_F1': Micro_average_F1,
                        'test_result': ans_mat.tolist(), 'BO_F1': BO_F1})

            if (e + 1) % 10000 == 0:
                torch.save(net.state_dict(), 'model/e{}_L{:.4f}_micF{:.4f}_{}.pkl'
                           .format(e, loss.item(), Micro_average_F1, int(time.time())))
        train_log.append(log)

    f = open(f'model/log_{int(time.time())}.json', 'w')
    f.write(json.dumps(train_log))
    f.close()


if __name__ == '__main__':
    train('model/e19999_L0.0133_micF0.9848_1621254511.pkl', 20000, learning_rate=0.025)
