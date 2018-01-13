#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import pickle
import os
import json


def load_map_file(map_filename):
    vocab, tag_to_id, id_to_tag = pickle.load(open(map_filename, 'rb'))
    return vocab, tag_to_id, id_to_tag


def seg_to_tfrecords(text_file, out_dir, map_file, out_name, indexs=[0, 1]):
    out_filename = os.path.join(out_dir, out_name + ".tfrecord")
    vocab, tag_to_id, id_to_tag = load_map_file(map_file)

    writer = tf.python_io.TFRecordWriter(out_filename)

    num_sample = 0
    all_oov = 0
    total_word = 0
    with open(text_file) as f:
        sent = []
        for lineno, line in enumerate(f):
            line = line.strip("\n")
            if not line:
                if sent:
                    num_sample += 1
                    word_count, oov_count = create_one_seg_sample(writer, sent, vocab, tag_to_id)
                    total_word += word_count
                    all_oov += oov_count
                    sent = []
                    continue

            word_info = line.split("\t")
            word_info = [word_info[i] for i in indexs]
            sent.append(word_info)
    print("oov rate : %f" % (1.0 * oov_count / total_word))

    return num_sample


def create_one_seg_sample(writer, sent, char_to_id, tag_to_id):
    char_list = []
    seg_label_list = []
    oov_count = 0
    word_count = 0
    for word in sent:
        ch = word[0]
        label = word[1]
        word_count += 1
        if ch in char_to_id:
            char_list.append(char_to_id[ch])
        else:
            char_list.append(char_to_id["<OOV>"])
            oov_count += 1
        seg_label_list.append(tag_to_id[label])

    example = tf.train.SequenceExample()

    sent_len = MAX_LENGTH if len(sent) > MAX_LENGTH else len(seg_label_list)

    fl_labels = example.feature_lists.feature_list["labels"]
    for l in seg_label_list[:sent_len]:
        fl_labels.feature.add().int64_list.value.append(l)

    fl_tokens = example.feature_lists.feature_list["char_list"]
    for t in char_list[:sent_len]:
        fl_tokens.feature.add().int64_list.value.append(t)

    fl_sent_len = example.feature_lists.feature_list["sent_len"]
    for t in [sent_len]:
        fl_sent_len.feature.add().int64_list.value.append(t)

    writer.write(example.SerializeToString())
    return word_count, oov_count


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="train file path")
    parser.add_argument("--dev_file", required=True, help="dev file path")
    parser.add_argument("--map_file", required=True, help="map file path")
    parser.add_argument("--test_file", required=True, help="test file path")
    parser.add_argument("--out_dir", required=True, help=" out dir for tfrecord ")
    parser.add_argument("--size_file", required=True, help="the size file create by create map step")
    parser.add_argument("--tag_index", default=-1, type=int, help="column index for target label ")
    parser.add_argument("--max_length", default=100, type=int,help="max length of sent")
    args = parser.parse_args()

    MAX_LENGTH = args.max_length

    train_num = seg_to_tfrecords(args.train_file, args.out_dir, args.map_file, "train", [0, args.tag_index])
    dev_num = seg_to_tfrecords(args.dev_file, args.out_dir, args.map_file, "dev", [0, args.tag_index])
    test_num = seg_to_tfrecords(args.test_file, args.out_dir, args.map_file, "test", [0, args.tag_index])

    print("train sample : %d" % (train_num))
    print("dev sample :%d" % (dev_num))
    print("test sample : %d" % (dev_num))

    size_filename = args.size_file

    with open(size_filename, 'r') as f:
        size_obj = json.load(f)

    with open(os.path.join(size_filename), 'w') as f:
        size_obj['train_num'] = train_num
        size_obj['dev_num'] = dev_num
        size_obj['test_num'] = test_num
        json.dump(size_obj, f)
