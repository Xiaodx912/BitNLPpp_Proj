import logging
import re
import unicodedata
from abc import ABCMeta, abstractmethod
from collections import Counter

import numpy as np
from typing import Tuple, List
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils.DataAgent import DataAgent
from utils.util_func import split_str_by_regex

para_id_reg = re.compile(r'^(199801[0-3][0-9]-\d{2}-\d{3}-\d{3})$')


class EncInterface(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def encode(self, unique_id: str) -> np.ndarray:
        pass


class OneHotEncodedPara(object):
    def __init__(self, length: int, dim_n: int, uid: str):
        self.length = length
        self.dim_n = dim_n
        self.uid = uid
        self.index_array = np.empty(self.length + 2, dtype=np.int16)
        self.index_array[:] = -1

    def data(self):
        try:
            assert self.index_array.min(initial=self.dim_n) >= -1
            assert self.index_array.max(initial=-1) < self.dim_n
        except AssertionError:
            return None
        vector = np.zeros((self.length, 3, self.dim_n), dtype=np.int8)
        for i in range(0, self.length + 2):
            if self.index_array[i] != -1:
                if 0 < i < self.length + 1:
                    vector[i - 1][1][self.index_array[i]] = 1
                if i > 1:
                    vector[i - 2][2][self.index_array[i]] = 1
                if i < self.length:
                    vector[i][0][self.index_array[i]] = 1
        return vector.reshape(self.length, 3 * self.dim_n)


class BIOLabeledPara(object):
    def __init__(self, uid: str, label: list):
        self.uid = uid
        self.length = len(label)
        self.label_list = label

    def get_bo_array(self):
        resp = np.zeros(self.length, dtype=np.int8)
        for i in range(self.length):
            if self.label_list[i][0] == '_':
                resp[i] = 1
        return resp

    def get_bio_label(self,length=None):  # B=2 I=1 O=0
        resp = np.zeros(self.length if length is None else length, dtype=np.int64)
        for i in range(self.length):
            if self.label_list[i][0] == '_':
                resp[i] = 2 if self.label_list[i][1] == 'B' else 1
            else:
                resp[i] = 0
        if length is not None and length>self.length :
            resp[self.length:]=-1
        return resp

    def get_bio_array(self):
        resp = np.zeros((self.length, 3), dtype=np.float32)
        for i in range(self.length):
            if self.label_list[i][0] == '_':
                resp[i][2 if self.label_list[i][1] == 'B' else 1] = 1
            else:
                resp[i][0] = 1
        return resp


class OneHotEncoder(EncInterface):
    def __init__(self, da: DataAgent, dim_n: int = 5000, overload: float = 1.2):
        self.logger = logging.getLogger('OneHotEncoder')
        self.logger.setLevel(logging.DEBUG)
        self.word_list = []
        self.DA = da
        self.dim_n = dim_n
        self.load_n = int(self.dim_n * overload)
        self.logger.info('Making encode dict...')
        self.generate_dict_from_da()
        self.word_list.extend(['^', '$'])  # '^' as paragraph's start,'$' as end
        self.extra_dim = 2
        self.word_dict = {}
        for index in range(len(self.word_list)):
            self.word_dict[self.word_list[index]] = index

    def generate_dict_from_da(self):
        full_str = ' '.join(self.DA.full())
        neg_list, nt_word_list = split_str_by_regex(full_str, r'\[([^]]+)]nt')

        sign_reg = re.compile(r'^.+/w[a-z]*$')
        other_word_list = [i.split('/', 1)[0] for i in list(filter(lambda x: not sign_reg.match(x), neg_list))]

        negative_words = Counter(dict(Counter(other_word_list).most_common(self.load_n)))
        positive_words = Counter(dict(Counter(nt_word_list).most_common(self.load_n)))
        self.word_list = list(dict((positive_words | negative_words).most_common(self.dim_n)).keys())
        self.logger.info(f'Dict size: {len(self.word_list)}.')

    def find_word_index(self, word: str) -> int:
        return self.word_dict.get(word, -1)

    def __find_word_index_slow(self, word: str) -> int:
        try:
            pos = self.word_list.index(word)
            return pos
        except ValueError:
            return -1

    def encode(self, unique_id: str) -> np.ndarray:  # unique_id refer to DataAgent.word()
        vector = np.zeros((3, self.dim_n + self.extra_dim), dtype=np.int8, order='C')
        match_result = re.match(r'^(199801[0-3][0-9]-\d{2}-\d{3}-\d{3})-(\d{4})$', unique_id)
        assert match_result is not None
        self.logger.debug(f'Encoding word with uid {unique_id}')
        for offset in (-1, 0, 1):
            uid = '{}{:0>4}'.format(unique_id[:-4], int(match_result[2]) + offset)
            resp = self.DA.get_word(uid)
            assert resp is not None
            word, label = resp.strip().replace('[', '').split('/', 1)
            pos = self.find_word_index(word)
            self.logger.debug(f'{word} (Offset{offset}) :{pos}')
            vector[1 + offset][pos] = 1
        return vector.flatten()

    def para_fast_enc(self, para_id: str) -> Tuple[OneHotEncodedPara, BIOLabeledPara]:
        assert para_id_reg.match(para_id) is not None
        para_str = self.DA.main_data[para_id]
        para_words = para_str.split()
        encoded = OneHotEncodedPara(len(para_words), self.dim_n + self.extra_dim, para_id)
        encoded.index_array[0] = self.find_word_index('^')
        encoded.index_array[len(para_words) + 1] = self.find_word_index('$')
        label_list = []
        for i in range(1, len(para_words) + 1):
            word, label = para_words[i - 1].strip().replace('[', '').split('/', 1)
            encoded.index_array[i] = self.find_word_index(word)
            label_list.append(label)
        labeled = BIOLabeledPara(para_id, label_list)
        return encoded, labeled


class WordEmbeddingEncoder(EncInterface):
    def __init__(self, da: DataAgent, path='data/ctb.50d.vec', mode='linear'):
        self.logger = logging.getLogger('WordEmbeddingEncoder')
        self.logger.setLevel(logging.DEBUG)
        self.DA = da
        self.vec_dict = {}
        self.zero_vec = np.zeros(50, dtype=np.float32)
        self.logger.info(f'Init with vector file {path}...')
        self.load_vec_from(path)
        self.logger.info(f'{len(self.vec_dict)} word vectors loaded in total.')
        self.mode = mode
        if self.mode == 'lstm':
            self.logger.info(f'Making word alphabet for RNN...')
            self.alphabet = {}
            self.gen_word_alphabet()
            self.logger.info(f'Alphabet size: {len(self.alphabet)}')

    def load_vec_from(self, vec_path):
        try:
            self.logger.debug(f'Try loading "{vec_path}"')
            f = open(vec_path, 'r', encoding='utf8')
            full_data = f.read().split('\n')
            f.close()
            self.logger.debug('Word vector file loaded.')
        except FileNotFoundError:
            self.logger.error(f'"{vec_path}" not exist!')
            return
        for i, line in enumerate(full_data):
            if (i + 1) % 50000 == 0 or i + 1 == len(full_data):
                print(f'\rLine# {i + 1}/{len(full_data)}', end='\n' if i + 1 == len(full_data) else '')
            if len(line) == 0:
                continue
            vec_str = line.split()
            vec = np.empty(50, dtype=np.float32)
            for i in range(50):
                vec[i] = float(vec_str[i + 1])
            self.vec_dict[vec_str[0]] = vec

    def encode(self, unique_id: str) -> np.ndarray:
        match_result = re.match(r'^(199801[0-3][0-9]-\d{2}-\d{3}-\d{3})-(\d{4})$', unique_id)
        assert match_result is not None
        self.logger.debug(f'Encoding word with uid {unique_id}')
        vec = []
        for offset in (-1, 0, 1):
            uid = '{}{:0>4}'.format(unique_id[:-4], int(match_result[2]) + offset)
            resp = self.DA.get_word(uid)
            assert resp is not None
            word, label = resp.strip().replace('[', '').split('/', 1)
            if label in ['0', '1']:
                vec.append(self.vec_dict['</s>'])
            else:
                vec.append(self.vec_dict.get(word, self.zero_vec))
        return np.concatenate(vec, axis=0)

    def para_fast_enc(self, para_id: str):
        assert para_id_reg.match(para_id) is not None
        para_str = self.DA.main_data[para_id]
        para_words = para_str.split()
        resp = np.zeros((3, len(para_words), 50), dtype=np.float32)
        label_list = []
        for i, word in enumerate(para_words):
            word_str, label = word.strip().replace('[', '').split('/', 1)
            resp[1][i] = self.vec_dict.get(word_str, self.vec_dict['-unknown-'])
            label_list.append(label)
        resp[0][1:] = resp[1][:-1]
        resp[2][:-1] = resp[1][1:]
        resp[0][0] = resp[2][-1] = self.vec_dict['</s>']
        return resp.swapaxes(0, 1).reshape(len(para_words), 150), BIOLabeledPara(para_id, label_list)

    def gen_word_alphabet(self):
        full_list = ' '.join(self.DA.full()).split()
        word_list = [i.split('/', 1)[0] for i in full_list]
        self.logger.debug(f'{len(word_list)} words total in text.')
        sorted_words = Counter(word_list).most_common()
        self.logger.debug(f'{len(sorted_words)} words after removing duplicates.')
        self.alphabet = {w[0]: i for i, w in enumerate([('__padding__', -1), ('</s>', -1)] + sorted_words)}
        self.vec_dict['__padding__'] = self.zero_vec

    def make_emb_layer(self):
        if self.mode != 'lstm' or len(self.alphabet) == 0:
            self.logger.error(f'Word alphabet not ready, exiting.')
            return None
        self.logger.info('Make embedding matrix with word alphabet.')
        emb_matrix = np.empty((len(self.alphabet), 50))
        for word, index in self.alphabet.items():
            emb_matrix[index] = self.vec_dict.get(word, self.vec_dict['-unknown-'])
            if word in self.vec_dict:
                emb_matrix[index] = self.vec_dict[word]
            elif unicodedata.normalize('NFKC', word) in self.vec_dict:
                emb_matrix[index] = self.vec_dict[unicodedata.normalize('NFKC', word)]
            else:
                emb_matrix[index] = self.vec_dict['-unknown-']
        self.logger.info('Embedding matrix is ready.')
        return emb_matrix

    def para_to_idx(self, para_id: str):
        assert para_id_reg.match(para_id) is not None
        para_str = self.DA.main_data[para_id]
        para_words = para_str.split()
        idx = np.empty(len(para_words), dtype=np.int32)
        label_list = []
        for i, word in enumerate(para_words):
            word_str, label = word.strip().replace('[', '').split('/', 1)
            idx[i] = self.alphabet.get(word_str, 0)
            label_list.append(label)
        return idx, BIOLabeledPara(para_id, label_list)

    def batch_to_packed_idx(self, para_batch: List[str]):
        padded = np.zeros((len(para_batch), self.DA.max_length), dtype=np.int32)
        length = np.empty(len(para_batch), dtype=np.int32)
        label_pack_list = []
        for i, para in enumerate(para_batch):
            idx_arr, label_pack = self.para_to_idx(para)
            length[i] = idx_arr.size
            padded[i][0:length[i]] = idx_arr[:]
            label_pack_list.append(label_pack)
        padded = torch.from_numpy(padded)
        return pack_padded_sequence(padded, length, batch_first=True, enforce_sorted=False), label_pack_list
