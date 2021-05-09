import re

import torch


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


def get_range_tensor(encoder, uid_list: list):
    _x_list = []
    _y_list = []
    for _i in range(len(uid_list)):
        _uid = uid_list[_i]
        if (_i + 1) % 1000 == 0:
            print(f'\r{_i + 1}/{len(uid_list) + 1}', end=' ')
        x, y = encoder.para_fast_enc(_uid)
        _x_list.append(torch.from_numpy(x.data()))
        _y_list.append(torch.from_numpy(y.get_bo_array()))
    print(f'\r{len(uid_list) + 1}/{len(uid_list) + 1}')
    return torch.cat(_x_list), torch.cat(_y_list)
