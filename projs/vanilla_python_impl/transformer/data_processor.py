import os
import copy
import random
import collections
import numpy as np


class Vocab:
    """Vocabulary for text."""

    def __init__(self, tokens, idx_to_token, token_to_idx, unk_index, min_freq=0):
        self.unk_index = unk_index

        # sort by freq
        counter = self.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        self.idx_to_token = copy.deepcopy(idx_to_token)
        self.token_to_idx = copy.deepcopy(token_to_idx)

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break

            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    @staticmethod
    def count_corpus(tokens):
        """先对输入转换成一个 1D list, 每个元素是一个 word 或一个 char, 再做频数统计"""
        # here tokens is a 1D or 2D list
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # unwrap
            tokens = [token for line in tokens for token in line]

        return collections.Counter(tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk_index)

        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (tuple, list)):
            return self.idx_to_token[indices]

        return [self.idx_to_token[index] for index in indices]

    @property
    def token_freqs(self):
        return self._token_freqs


class DataProcessor:
    """
    Data processor.
    Using the eng-fra dataset.

    Reference:
        1. https://zh.d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html
    """

    def __init__(self):
        self.DATASET_NAME = 'eng-fra.txt'

        self.PAD_TOKEN = '<pad>'
        self.BOS_TOKEN = '<bos>'
        self.EOS_TOKEN = '<eos>'
        self.UNK_TOKEN = '<unk>'

        self.PAD_INDEX = 0
        self.BOS_INDEX = 1
        self.EOS_INDEX = 2
        self.UNK_INDEX = 3

        self._idx_to_token = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        self._token_to_idx = {token: idx for idx, token in enumerate(self._idx_to_token)}

    def prepare_data(self, path, batch_size, test_ratio=0.1, min_freq=2, num_examples=None):
        """
        Prepare eng-fra dataset.
        ------------------------
            Args:
                num_examples: the number of examples used for training and testing (all if None)
        """
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.min_freq = min_freq
        self.num_examples = num_examples

        text = self._preprocess(self._read_data(path))
        source, target = self._tokenize(text, num_examples)

        # build vocab
        self.src_vocab = Vocab(source, self._idx_to_token, self._token_to_idx, self.UNK_INDEX, min_freq)
        self.tgt_vocab = Vocab(target, self._idx_to_token, self._token_to_idx, self.UNK_INDEX, min_freq)

        # to index
        source = self.src_vocab[source]
        target = self.tgt_vocab[target]

        train_src, train_tgt, test_src, test_tgt = self._train_test_split(source, target, test_ratio)

        # build dataset in batches
        self.train_src, self.train_tgt = self._build_dataset(train_src, train_tgt, batch_size)
        self.test_src, self.test_tgt = self._build_dataset(test_src, test_tgt, batch_size)

        self.num_train_batches = len(self.train_src)
        self.num_test_batches = len(self.test_src)

        # src_mask: (batch_size, 1, max_len_in_this_batch)
        # tgt_mask: (batch_size, max_len_in_this_batch, max_len_in_this_batch)
        self.train_src_mask = self.get_pad_mask(self.train_src)
        self.test_src_mask = self.get_pad_mask(self.test_src)
        self.train_tgt_mask = self._and_pad_sub_mask(self.get_pad_mask(self.train_tgt),
                                                     self.get_sub_mask(self.train_tgt))
        self.test_tgt_mask = self._and_pad_sub_mask(self.get_pad_mask(self.test_tgt),
                                                    self.get_sub_mask(self.test_tgt))

        self._print_dataset_info()

    def train_iter(self):
        """Training iterator."""
        idx = list(range(self.num_train_batches))
        random.shuffle(idx)

        for i in range(self.num_train_batches):
            yield (self.train_src[idx[i]], self.train_src_mask[idx[i]]), (self.train_tgt[idx[i]],
                                                                          self.train_tgt_mask[idx[i]])

    def test_iter(self):
        """Testing iterator."""
        idx = list(range(self.num_test_batches))
        random.shuffle(idx)

        for i in range(self.num_test_batches):
            yield (self.test_src[idx[i]], self.test_src_mask[idx[i]]), (self.test_tgt[idx[i]],
                                                                        self.test_tgt_mask[idx[i]])

    def _read_data(self, path):
        """
        Read in dataset.

        Return: str
        """
        with open(os.path.join(path, self.DATASET_NAME), 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _preprocess(self, text):
        """
        Preprocess eng-fra dataset.

        Return: str
        """
        def no_space(char, prev_char):
            return char in set(',.!?') and prev_char != ' '

        # 使用空格替换不间断空格
        # 使用小写字母替换大写字母
        text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

        # 在单词和标点符号之间插入空格, 从而之后可之间用 split 按空格对句子进行分词
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]

        return ''.join(out)

    def _tokenize(self, text, num_examples=None):
        """
        Tokenize eng-fra dataset.
        -------------------------
            Returns:
                two list
        """
        source, target = [], []

        for i, line in enumerate(text.split('\n')):
            if num_examples and i == num_examples:
                break

            parts = line.split('\t')
            if len(parts) == 2:
                source.append(parts[0].split(' '))
                target.append(parts[1].split(' '))

        return source, target

    def _train_test_split(self, src, tgt, test_ratio=0.1):
        idx = list(range(len(src)))
        random.shuffle(idx)
        len_test = int(len(src) * test_ratio)

        train_src, train_tgt, test_src, test_tgt = [], [], [], []

        for i in range(len_test):
            test_src.append(src[idx[i]])
            test_tgt.append(tgt[idx[i]])

        for i in range(len(src) - len_test):
            train_src.append(src[idx[len_test + i]])
            train_tgt.append(tgt[idx[len_test + i]])

        return train_src, train_tgt, test_src, test_tgt

    def _build_dataset(self, src, tgt, batch_size):
        """
        Return:
            ret_src: a list, each element is a np.array with shape (batch_size, max_len_in_this_batch)
        """
        # since dataset has been shuffled in train_test_split, we won't shuffle here
        for i in range(len(src)):
            src[i] = [self.BOS_INDEX] + src[i] + [self.EOS_INDEX]

        for i in range(len(tgt)):
            tgt[i] = [self.BOS_INDEX] + tgt[i] + [self.EOS_INDEX]

        # this will drop some samples
        num_batches = len(src) // batch_size
        ret_src, ret_tgt = [], []

        for i in range(num_batches):
            src_max_len, tgt_max_len = 0, 0

            for j in range(batch_size):
                src_max_len = max(src_max_len, len(src[i * batch_size + j]))
                tgt_max_len = max(tgt_max_len, len(tgt[i * batch_size + j]))

            for j in range(batch_size):
                src[i * batch_size + j] += [self.PAD_INDEX] * (src_max_len - len(src[i * batch_size + j]))
                tgt[i * batch_size + j] += [self.PAD_INDEX] * (tgt_max_len - len(tgt[i * batch_size + j]))

            ret_src.append(np.array(src[i * batch_size:(i + 1) * batch_size]))
            ret_tgt.append(np.array(tgt[i * batch_size:(i + 1) * batch_size]))

        return ret_src, ret_tgt

    def get_pad_mask(self, x):
        """
        Return pad mask.
        ----------------
            Args:
                x: a list, each element is a np.array with shape (batch_size, max_len_in_this_batch)
        """
        ret = []

        for e in x:
            ret.append((e != self.PAD_INDEX).astype(int)[:, np.newaxis, :])

        return ret

    def get_sub_mask(self, x):
        """
        Return sub mask, only used for tgt.
        ----------------
            Args:
                x: a list, each element is a np.array with shape (batch_size, max_len_in_this_batch)
        """
        ret = []

        for e in x:
            seq_len = e.shape[1]
            subsequent_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(int)
            ret.append(np.logical_not(subsequent_mask))

        return ret

    def _and_pad_sub_mask(self, pad, sub):
        """
        And operator on pad and sub mask, only used for tgt.
        """
        ret = []

        for i in range(len(pad)):
            ret.append(pad[i] & sub[i])

        return ret

    def _print_dataset_info(self):
        print('------Dataset Info------')
        print(f'Dataset Name: {self.DATASET_NAME}')
        print(f'Batch Size: {self.batch_size}, Min Freq: {self.min_freq}, Test Ratio: {self.test_ratio}')
        print(f'Source Vocab Size: {len(self.src_vocab)}, Target Vocab Size: {len(self.tgt_vocab)}')
        print(f'Number of Train Batches: {self.num_train_batches}, Number of Test Batches: {self.num_test_batches}')
        print('------------------------', end='\n\n')


if __name__ == '__main__':
    dp = DataProcessor()
    dp.prepare_data('../../../data/vanilla_python_impl/transformer', 64)

    for i, x in enumerate(dp.train_iter()):
        s, t = x
        src, src_mask = s
        tgt, tgt_mask = t
        print(f'{i}: {src.shape}, {src_mask.shape}, {tgt.shape}, {tgt_mask.shape}')
