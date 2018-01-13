#!/usr/bin/env python
# -*-coding:utf-8-*-

import os
import pickle
import argparse
import json


OOV_STR = "<OOV>"
PAD_STR = "<PAD>"


def create_vocab(embeding_file):

    vocab_dict = {}
    vocab_dict[PAD_STR] = 0
    vocab_dict[OOV_STR] = 1

    f = open(embeding_file, errors="ignore")
    m, n = f.readline().split(" ")
    n = int(n)
    m = int(m)
    print("preembeding size : %d"%(m))

    for i, line in enumerate(f):
        word = line.split()[0]
        if not word:
            continue
        if word not in vocab_dict:
            vocab_dict[word] = len(vocab_dict)
    print("vocab size : %d" % len(vocab_dict))
    return vocab_dict


def tag_to_map(train_file, tag_index=-1):
    f = open(train_file)
    tag_to_id = {}

    for i, line in enumerate(f):
        line = line.strip("\n")
        if not line:
            continue
        data = line.split("\t")
        # todo hand this in train file

        tag = data[tag_index]
        if tag not in tag_to_id:
            tag_to_id[tag] = len(tag_to_id)

    id_to_tag = {v: k for k, v in tag_to_id.items()}

    print("tag num in %s: %d "%(train_file, len(tag_to_id)))
    return tag_to_id, id_to_tag


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="train file path")
    parser.add_argument("--embeding_file", required=True, help="embeding file path")
    parser.add_argument("--map_file", required=True, help="the out dir of map file")
    parser.add_argument("--tag_index", required=False, type=int, default=-1, help="the column num of taget tag in train_file ")
    parser.add_argument("--size_file", required=True, help="save size to")
    args = parser.parse_args()

    # if not os.path.exists(args.out_dir):
    #     os.mkdir(args.out_dir)

    with open(args.map_file, 'wb') as f:
        vocab = create_vocab(embeding_file=args.embeding_file)
        tag_to_id, id_to_tag = tag_to_map(train_file=args.train_file, tag_index=args.tag_index)
        print("tag map result: ")
        print(tag_to_id)
        pickle.dump((vocab, tag_to_id, id_to_tag), f)
        vocab_size = len(vocab)
        num_class = len(tag_to_id)
        print("vocab size :%d, num of tag : %d"%(vocab_size, num_class))
        size_file = open(args.size_file, 'w')
        json.dump({"vocab_size": vocab_size, "num_tag":num_class}, size_file)
