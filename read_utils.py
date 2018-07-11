
from tqdm import tqdm
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import itertools
import pickle
import random

def samples_clearn(samples):
    cl_samples = [sample for sample in samples if sample[4]==1]
    return cl_samples

def val_samples_generator(samples):
    val_g = []
    n = len(samples)
    batchsize = 1000
    for i in range(0,n,batchsize):
        batch_samples = samples[i:i + batchsize]
        batch_q = []
        batch_q_len = []
        batch_r = []
        batch_r_len = []
        batch_y = []
        for sample in batch_samples:
            batch_q.append(sample[0])
            batch_q_len.append(sample[1])
            batch_r.append(sample[2])
            batch_r_len.append(sample[3])
            batch_y.append(sample[4])

        batch_q = np.array(batch_q)
        batch_q_len = np.array(batch_q_len)
        batch_r = np.array(batch_r)
        batch_r_len = np.array(batch_r_len)
        batch_y = np.array(batch_y)
        val_g.append( (batch_q,batch_q_len,batch_r,batch_r_len,batch_y))
    return val_g

def test_samples_generator(samples):
    batch_q = []
    batch_q_len = []
    batch_r = []
    batch_r_len = []
    for sample in samples:
        batch_q.append(sample[0])
        batch_q_len.append(sample[1])
        batch_r.append(sample[2])
        batch_r_len.append(sample[3])

    batch_q = np.array(batch_q)
    batch_q_len = np.array(batch_q_len)
    batch_r = np.array(batch_r)
    batch_r_len = np.array(batch_r_len)

    return batch_q,batch_q_len,batch_r,batch_r_len

def batch_generator(samples, batchsize):
    '''产生训练batch样本'''
    n_samples = len(samples)
    n_batches = int(n_samples/batchsize)
    n = n_batches * batchsize
    while True:
        random.shuffle(samples)  # 打乱顺序
        for i in range(0, n, batchsize):
            batch_samples = samples[i:i+batchsize]
            batch_q = []
            batch_q_len = []
            batch_r = []
            batch_r_len = []
            batch_y = []
            for sample in batch_samples:
                batch_q.append(sample[0])
                batch_q_len.append(sample[1])
                batch_r.append(sample[2])
                batch_r_len.append(sample[3])
                batch_y.append(sample[4])

            batch_q = np.array(batch_q)
            batch_q_len = np.array(batch_q_len)
            batch_r = np.array(batch_r)
            batch_r_len = np.array(batch_r_len)
            batch_y = np.array(batch_y)
            yield batch_q,batch_q_len,batch_r,batch_r_len,batch_y

def load_origin_data(data_path):
    with open(data_path, 'r',encoding='utf-8') as f:
        line = f.readline()
        QAs = []
        text = ''
        while line:
            line = line.strip()
            qas = line.split('\t')
            assert len(qas)==3,'error qas len:%s' % len(qas)

            text += qas[0]+qas[1]
            QAs.append(tuple(qas))

            line = f.readline()
        print('问答对总数:%s' % len(QAs))
        return QAs, text


class TextConverter(object):
    def __init__(self, text=None, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)
            print('字符数量：%s ' % len(vocab))
            # max_vocab_process
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            vocab_count_list = []
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, obj, filename):
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)

    def load_obj(self,filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            return obj


    def Q_to_arr(self, query ,n_steps):
        query_arr = self.text_to_arr(query)
        query_len = len(query)

        # 补全
        if query_len < n_steps:
            query_arr = np.append(query_arr, [len(self.vocab)] * (n_steps - query_len))
        else:
            query_arr = query_arr[:n_steps]
            query_len = n_steps
        return query_arr,np.array(query_len)

    def QAs_to_arrs(self, QAs, n_steps):
        QA_arrs = []
        last_num = len(self.vocab)
        for query, response,label in QAs:
            query_len = len(query)
            response_len = len(response)

            # text to arr
            query_arr = self.text_to_arr(query)
            response_arr = self.text_to_arr(response)

            # 补全
            if query_len<n_steps:
                query_arr = np.append(query_arr,[last_num]*(n_steps-query_len))
            else:
                query_arr = query_arr[:n_steps]
                query_len = n_steps
            if response_len<n_steps:
                response_arr = np.append(response_arr, [last_num] * (n_steps - response_len))
            else:
                response_arr = response_arr[:n_steps]
                response_len = n_steps

            QA_arrs.append([query_arr,query_len,response_arr,response_len, float(label)])

        return QA_arrs

    def testQAs_to_arrs(self, QAs, n_steps):
        QA_arrs = []
        last_num = len(self.vocab)
        for id,query, response in QAs:
            query_len = len(query)
            response_len = len(response)

            # text to arr
            query_arr = self.text_to_arr(query)
            response_arr = self.text_to_arr(response)

            # 补全
            if query_len<n_steps:
                query_arr = np.append(query_arr,[last_num]*(n_steps-query_len))
            else:
                query_arr = query_arr[:n_steps]
                query_len = n_steps
            if response_len<n_steps:
                response_arr = np.append(response_arr, [last_num] * (n_steps - response_len))
            else:
                response_arr = response_arr[:n_steps]
                response_len = n_steps

            QA_arrs.append([query_arr,query_len,response_arr,response_len])

        return QA_arrs


if __name__=="__main__":
    pro_data = TextConverter('word','data','process_data1',26)

    # print(pro_data.PAD_INT)

