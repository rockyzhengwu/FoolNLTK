#!/usr/bin/env python
#-*-coding:utf-8-*-

import math
import random
import pickle
import re
import json
import tensorflow as tf


def load_size_file(size_filename):
    with open(size_filename, 'r') as f:
        print(size_filename)
        num_obj = json.load(f)
        return num_obj



def load_sample_size(filename):
    with open(filename) as f:
        train_num, dev_num, test_num = map(int, f.readline().split(","))
    return train_num, dev_num, test_num



def reverse_dict(d):
    return {v: k for k, v in d.items()}


def load_map_file(path, ):
    vocab, tag_to_id, id_to_tag = pickle.load(open(path, 'rb'))
    return vocab, tag_to_id, id_to_tag


def load_sentences(path, indexs):
    f = open(path)
    sentences = []
    sent = []
    for i, line in enumerate(f):
        line = line.strip("\n")
        if not line:
            if sent:
                sentences.append(sent)
                sent = []
            continue
        word_info = line.split("\t")
        word_info = [word_info[i] for i in indexs]
        sent.append(word_info)
    return sentences


def create_dataset(data_list, word_to_id, tag_to_id):
    data_set = []
    oov_count = 0
    total_count = 0
    space_word_count = 0
    for data in data_list:
        sent = []
        tag = []
        for w in data:
            total_count += 1
            if w[0] in word_to_id:
                sent.append(word_to_id[w[0]])
            elif w[0] in ["\t", "\n", "\r", " "]:
                space_word_count += 1
                sent.append(word_to_id["</s>"])
            else:
                sent.append(word_to_id["<OOV>"])
                oov_count += 1
            tag.append(tag_to_id[w[-1]])
        data_set.append([sent, tag])
    print("space word count: %d "%(space_word_count))
    print("dataset oov count:%d percent: %f"%(oov_count, 1.0 * oov_count / total_count))
    return data_set

en_p = re.compile('[a-zA-Z]', re.U)
re_han = re.compile("([\u4E00-\u9FD5]+)")

def get_char_type(ch):
    """
    0, 汉字
    1, 英文字母
    2. 数字
    3. 其他
    """
    if re.match(en_p, ch):
        return 1
    elif re.match("\d+", ch):
        return 2
    elif re.match(re_han, ch):
        return 3
    else:
        return 4


def create_ner_dataset(data_list, word_to_id, tag_to_id, pos_to_id, seg_to_id):
    data_set = []
    oov_count = 0
    total_count = 0
    for data in data_list:
        sent = []
        tag = []
        seg_list = []
        pos_list = []
        char_type_list = []
        for w in data:
            total_count += 1
            if w[0] in word_to_id:
                sent.append(word_to_id[w[0]])
            else:
                sent.append(word_to_id["<OOV>"])
                oov_count += 1
            tag.append(tag_to_id[w[-1]])
            seg_label = w[1]
            seg_list.append(seg_to_id[seg_label])
            pos_list.append(pos_to_id[w[2]])
            char_type_list.append(get_char_type(w[0]))
        data_set.append([sent, seg_list, pos_list, char_type_list, tag])
    print("dataset oov count:%d percent: %f" % (oov_count, 1.0 * oov_count / total_count))
    return data_set



class BatchManager(object):

    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            targets.append(target + padding)
        return [strings, targets]


    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)

        for idx in range(self.len_data):
            yield self.batch_data[idx]



class NERBatchManager(object):

    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        strings = []
        segs = []
        pos_list = []
        targets = []
        char_types = []

        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, seg, pos, char_type, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            segs.append(seg + padding)
            pos_list.append(pos + padding)
            targets.append(target + padding)
            char_types.append(char_type + padding)
        return [strings, segs, pos_list, char_types, targets]


    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)

        for idx in range(self.len_data):
            yield self.batch_data[idx]

class SegBatcher(object):
    def __init__(self, record_file_name, batch_size, num_epochs=None):
        self._batch_size = batch_size
        self._epoch = 0
        self._step = 1.
        self.num_epochs = num_epochs
        self.next_batch_op = self.input_pipeline(record_file_name, self._batch_size, self.num_epochs)


    def example_parser(self, filename_queue):
        reader = tf.TFRecordReader()
        key, record_string = reader.read(filename_queue)

        features = {
            'labels': tf.FixedLenSequenceFeature([], tf.int64),
            'char_list': tf.FixedLenSequenceFeature([], tf.int64),
            'sent_len': tf.FixedLenSequenceFeature([], tf.int64),
        }

        _, example = tf.parse_single_sequence_example(serialized=record_string, sequence_features=features)
        labels = example['labels']
        char_list = example['char_list']
        sent_len = example['sent_len']
        return labels, char_list, sent_len

    def input_pipeline(self, filenames, batch_size,  num_epochs=None):
        filename_queue = tf.train.string_input_producer([filenames], num_epochs=num_epochs, shuffle=True)
        labels, char_list, sent_len = self.example_parser(filename_queue)

        min_after_dequeue = 10000
        capacity = min_after_dequeue + 12 * batch_size
        next_batch = tf.train.batch([labels, char_list, sent_len], batch_size=batch_size, capacity=capacity,
                                        dynamic_pad=True, allow_smaller_final_batch=True)
        return next_batch


