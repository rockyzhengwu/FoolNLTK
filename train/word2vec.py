#!/usr/bin/env python
# -*-coding=utf-8-*-

import numpy as np

class Word2vec(object):
    def __init__(self):
        self.wv = {}


    def load_w2v_array(self, path, id_to_word, is_binary=False):
        """

        :param path:
        :param vocab_index2word: vocab index to word
        :param is_binary:
        :return:
        """

        if not is_binary:
            f = open(path, errors="ignore")
            m, n = f.readline().split()
            dim = int(n)

            print("%s words dim : %s"% (m, n ))
            for  i, line in enumerate(f):
                line = line.strip("\n").strip().split(" ")
                word = line[0]
                vec  =[float(v) for v in line[1:]]
                if len(vec)!= dim:
                    continue

                self.wv[word] = vec

        vocab_size = len(id_to_word)
        embedding = []

        bound =  np.sqrt(6.0) / np.sqrt(vocab_size)
        word2vec_oov_count = 0

        for i in range(vocab_size):
            word = id_to_word.get(i)
            if word in self.wv:
                embedding.append(self.wv.get(word))
            else:
                # todo 随机赋值为何?
                word2vec_oov_count += 1
                embedding.append(np.random.uniform(-bound, bound, dim));

        print("word2vec oov count: %d"%(word2vec_oov_count,))
        return np.array(embedding)


