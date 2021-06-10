import re

import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence


def flatten(a):
    for each in a:
        if not isinstance(each, list):
            yield each
        else:
            yield from flatten(each)


def split_str_by_regex(s: str, regex: str) -> (list, list):
    r = re.compile(regex)
    match_list = r.findall(s)
    match_word_array = [i.split() for i in match_list]
    match_words = [i.split('/', 1)[0] for i in flatten(match_word_array)]
    remain_str = r.sub('', s)
    return remain_str.split(), match_words


def get_range_tensor(encoder, uid_list: list, label_type='bo'):
    _x_list = []
    _y_list = []
    for _i in range(len(uid_list)):
        _uid = uid_list[_i]
        if (_i + 1) % 1000 == 0:
            print(f'\r{_i + 1}/{len(uid_list)}', end=' ')
        x, y = encoder.para_fast_enc(_uid)
        if not isinstance(x, np.ndarray):
            x = x.data()
        _x_list.append(torch.from_numpy(x))
        if label_type == 'bo':
            _y_list.append(torch.from_numpy(y.get_bo_array()))
        else:
            if label_type == 'bio_arr':
                _y_list.append(torch.from_numpy(y.get_bio_array()))
            else:
                _y_list.append(torch.from_numpy(y.get_bio_label()))
    print(f'\r{len(uid_list)}/{len(uid_list)}')
    return torch.cat(_x_list), torch.cat(_y_list)


def calc_F1(TP, FN, FP):
    if TP == 0:
        return 0
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    return 2 * r * p / (r + p)


def calc_masked_result(packed_vec, bio_labels, network: torch.nn.Module, device):
    packed_pred = network(packed_vec.to(device))
    pred_arr = pad_packed_sequence(packed_pred, batch_first=True)[0]
    label_arr = torch.tensor([label.get_bio_label(length=pred_arr.shape[1]) for label in bio_labels], device=device)
    mask = torch.ne(label_arr, -1)  # “mask = (label_arr != -1)” will confuse IDE's type prediction
    label_arr = label_arr[mask]
    pred_arr = pred_arr.masked_select(mask.unsqueeze(2).expand(-1, -1, network.fc.out_features)) \
        .contiguous().view(-1, network.fc.out_features)
    assert pred_arr.shape == (label_arr.shape[0], network.fc.out_features)
    return pred_arr, label_arr


def make_ans_eval_mat(pred, label):
    ans_mat = np.zeros((3, 3), dtype=np.int64)
    for ref_label in [0, 1, 2]:
        for pred_label in [0, 1, 2]:
            ans_mat[ref_label][pred_label] = ((label == ref_label) & (pred == pred_label)).sum()
    return ans_mat


def calc_tri_classification_f1(ans_mat):
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
    return {'Mac': Macro_average_F1_list.mean(), 'Mic': Micro_average_F1, 'BO': BO_F1}
