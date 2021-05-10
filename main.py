import json
import logging
import time

import torch

from utils.DataAgent import DataAgent
from utils.Encoder import OneHotEncoder
from utils.util_func import get_range_tensor

logging.basicConfig(format='%(asctime)s - %(levelname)s[%(name)s] - %(message)s')
logger = logging.getLogger('Main')
logger.setLevel(logging.DEBUG)

logger.debug(f'Torch ver: {torch.__version__}\tCUDA Accelerate: {torch.cuda.is_available()}')


def train(path=None, e_start=0, e_size=1000, learning_rate=0.02):
    da = DataAgent()
    enc = OneHotEncoder(da)

    logger.info('Loading data...')
    training_data = da.range_uid('19980101', '19980120')
    validation_data = da.range_uid('19980121', '19980125')
    test_data = da.range_uid('19980126', '19980131')

    logger.info('Making tensor from data...')
    train_x_tensor, train_y_ref = get_range_tensor(enc, training_data)
    test_x_tensor, test_y_ref = get_range_tensor(enc, validation_data)
    logger.info('Casting tensor into float...')
    train_x_tensor, train_y_ref = train_x_tensor.float(), train_y_ref.float()
    test_x_tensor, test_y_ref = test_x_tensor.float(), test_y_ref.float()

    class LinearRegression(torch.nn.Module):
        def __init__(self):
            super(LinearRegression, self).__init__()
            self.fc = torch.nn.Linear(15006, 1)

        def forward(self, x):
            out = self.fc(x)
            out = torch.sigmoid(out)
            return out

    logger.info('Building network')
    net = LinearRegression()
    criterion = torch.nn.BCELoss()
    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)  # Adam优化
    scheduler = torch.optim.lr_scheduler. \
        ReduceLROnPlateau(optim, mode='min', verbose=True, patience=10, threshold_mode='abs', threshold=0.00002,
                          factor=0.8, cooldown=50, min_lr=0.001)
    batch_size = 1 << 16

    if path:
        logger.info(f'Loading parameter from {path}')
        net.load_state_dict(torch.load(path))

    train_log = []
    logger.info('Start training network.')
    for e in range(e_start, e_start + e_size):
        net.train()
        y_pred = net(train_x_tensor).flatten()
        loss = criterion(y_pred, train_y_ref)
        scheduler.step(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()
        log = {'epoch': e, 'loss': float(loss.item()), 'LR': float(optim.state_dict()['param_groups'][0]['lr'])}
        if (e + 1) % 5 == 0:
            net.eval()
            test_ans = net(test_x_tensor).flatten()
            TP = ((test_y_ref == 1) & (test_ans > 0.5)).sum()
            FN = ((test_y_ref == 0) & (test_ans > 0.5)).sum()
            FP = ((test_y_ref == 1) & (test_ans < 0.5)).sum()
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            F1 = 2 * r * p / (r + p)
            logger.info("Epoch:{},Loss:{:.4f},F1:{:.4f},LR:{:.4f}"
                        .format(e + 1, loss.item(), F1, optim.state_dict()['param_groups'][0]['lr']))
            log.update({'TP': float(TP), 'FN': float(FN), 'FP': float(FP), 'F1': float(F1)})
            if (e + 1) % 100 == 0:
                torch.save(net.state_dict(),
                           'model/e{}_L{:.4f}_F{:.4f}_{}.pkl'.format(e, loss.item(), F1, int(time.time())))
        train_log.append(log)
    str_log = json.dumps(train_log)
    f = open(f'model/log_{int(time.time())}.json', 'w')
    f.write(str_log)
    f.close()
    return train_log


if __name__ == '__main__':
    train_log = train(path='model/e1999_L0.0275_F0.6169_1620659839.pkl', e_start=2000, e_size=500, learning_rate=0.02)
