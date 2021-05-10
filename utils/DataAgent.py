import logging
import re

from typing import List


class DataAgent:
    def __init__(self, data_path='data/1998-01-2003版-带音.txt'):
        self.logger = logging.getLogger('DataAgent')
        self.logger.setLevel(logging.DEBUG)
        self.logger.info(f'Data Agent init with path "{data_path}"')
        self.main_data = {}
        self.load_from_txt(data_path)

    def load_from_txt(self, txt_path):
        try:
            self.logger.debug(f'Try loading "{txt_path}"')
            f = open(txt_path, 'r')
            full_data = f.read().split('\n')
            f.close()
        except FileNotFoundError:
            self.logger.error(f'"{txt_path}" not exist!')
            return
        self.logger.debug(f'{len(full_data)} lines total.')
        uid_matcher = re.compile(r'^199801([0-3][0-9])-(\d{2})-(\d{3})-(\d{3})/m\s+')
        nt_matcher = re.compile(r'\[([^\[\]]+)]nt')
        for paragraph in full_data:
            r_match = uid_matcher.match(paragraph)
            if r_match:
                for nt_phrase in nt_matcher.findall(paragraph):
                    nt_bio = nt_phrase.replace('/', '/_I_').replace('/_I_', '/_B_', 1)
                    paragraph = paragraph.replace(nt_phrase, nt_bio)
                self.main_data[r_match[0][:-4]] = paragraph[23:].strip()
        self.logger.debug(f'{len(self.main_data)} lines useful.')

    def range(self, l_lim, r_lim) -> List[str]:
        resp = []
        for k, v in self.main_data.items():
            if int(k[:8]) in range(int(l_lim), int(r_lim) + 1):
                resp.append(v)
        self.logger.info(f'Find {len(resp)} paragraph between {l_lim} and {r_lim}.')
        return resp

    def range_uid(self, l_lim, r_lim) -> List[str]:
        resp = []
        for k in self.main_data.keys():
            if int(k[:8]) in range(int(l_lim), int(r_lim) + 1):
                resp.append(k)
        self.logger.info(f'Find {len(resp)} paragraph\'s uid between {l_lim} and {r_lim}.')
        return resp

    def full(self) -> list:
        return list(self.main_data.values())

        # unique_id     19980101-   01-     001-        001-        0001
        #               date        page    article     paragraph   word

    def get_word(self, unique_id: str):
        match_result = re.match(r'^(199801[0-3][0-9]-\d{2}-\d{3}-\d{3})-(\d{4})$', unique_id)
        if match_result is None:
            self.logger.error(f'Invalid unique ID {unique_id}!')
            return None
        try:
            paragraph = self.main_data[match_result[1]]
        except KeyError:
            self.logger.error(f'Page id {match_result[1]} out of range!')
            return None
        word_list = paragraph.split()
        word_n = int(match_result[2])
        if word_n in range(1, len(word_list) + 1):
            return word_list[word_n - 1]
        if word_n == 0:
            return '^/0'
        if word_n == len(word_list) + 1:
            return '$/1'
        self.logger.error(f'Word id {word_n} out of {match_result[1]}\'s range!(paragraph length {len(word_list)})')
        return None
