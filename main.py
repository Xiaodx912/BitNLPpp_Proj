import json
import logging

import torch

from utils.DataAgent import DataAgent
from utils.Encoder import OneHotEncoder
from utils.util_func import get_range_tensor

logging.basicConfig(format='%(asctime)s - %(levelname)s[%(name)s] - %(message)s')
logger = logging.getLogger('Main')
logger.setLevel(logging.DEBUG)

logger.debug(f'Torch ver: {torch.__version__}\tCUDA Accelerate: {torch.cuda.is_available()}')

DA = DataAgent()
enc = OneHotEncoder(DA)

logger.info('Loading data...')
training_data = DA.range_uid('19980101', '19980120')
validation_data = DA.range_uid('19980121', '19980125')
test_data = DA.range_uid('19980126', '19980131')

logger.info('Making tensor from data...')
train_x_tensor, train_y_ref = get_range_tensor(enc, training_data)
test_x_tensor, test_y_ref = get_range_tensor(enc, validation_data)
logger.info('Casting tensor into float...')
train_x_tensor, train_y_ref = train_x_tensor.float(), train_y_ref.float()
test_x_tensor, test_y_ref = test_x_tensor.float(), test_y_ref.float()


class LR(torch.nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.fc = torch.nn.Linear(15006, 1)

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out


logger.info('Building network')
net = LR()
criterion = torch.nn.BCELoss()
optim = torch.optim.Adam(net.parameters(), lr=0.02)  # Adam优化
epochs = 1000
batch_size = 1 << 16


def main(path=None,e_start=0):
    train_log = []
    logger.info('Start training network.')
    for e in range(epochs):
        net.train()
        y_pred_list = []
        # for i in range(ceil(len(train_x_tensor) / batch_size)):
        #   x = train_x_tensor[batch_size * i:batch_size * (i + 1)]
        #   ans = net(x.float())
        #   y_pred_list.append(ans)
        # y_pred = torch.cat(y_pred_list).flatten()
        y_pred = net(train_x_tensor).flatten()
        loss = criterion(y_pred, train_y_ref)
        optim.zero_grad()
        loss.backward()
        optim.step()
        torch.cuda.empty_cache()
        log = {'epoch': e, 'loss': float(loss.item())}
        if (e + 1) % 5 == 0:
            net.eval()
            test_ans = net(test_x_tensor).flatten()
            TP = ((test_y_ref == 1) & (test_ans > 0.5)).sum()
            FN = ((test_y_ref == 0) & (test_ans > 0.5)).sum()
            FP = ((test_y_ref == 1) & (test_ans < 0.5)).sum()
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            F1 = 2 * r * p / (r + p)
            logger.info("Epoch:{},Loss:{:.4f},F1：{:.4f}".format(e + 1, loss.item(), F1))
            log.update({'TP': float(TP), 'FN': float(FN), 'FP': float(FP), 'F1': float(F1)})
        train_log.append(log)
        if (e + 1) % 100 == 0:
            torch.save(net.state_dict(), 'model/e{}_L{:.4f}_F{:.4f}.pkl'.format(e, loss.item(), F1))
    str_log = json.dumps(train_log)
    f = open('model/train.log', 'w')
    f.write(str_log)
    f.close()
    return train_log


if __name__ == '__main__':
    main_log = main()

