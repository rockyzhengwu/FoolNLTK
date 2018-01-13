#!/usr/bin/env python
#-*-coding:utf-8-*-


import tensorflow as tf
import pickle
import numpy as np
from tensorflow.contrib.crf import viterbi_decode

def decode(logits, trans, sequence_lengths, tag_num):
    viterbi_sequences = []
    small = -1000.0
    start = np.asarray([[small] * tag_num + [0]])
    for logit, length in zip(logits, sequence_lengths):
        score = logit[:length]
        pad = small * np.ones([length, 1])
        logits = np.concatenate([score, pad], axis=1)
        logits = np.concatenate([start, logits], axis=0)
        viterbi_seq, viterbi_score = viterbi_decode(logits, trans)
        viterbi_sequences += viterbi_seq[1:]
    return viterbi_sequences


def load_map(path):
    with open(path, 'rb') as f:
        char_to_id,  tag_to_id, id_to_tag = pickle.load(f)
    return char_to_id, id_to_tag


def load_graph(path):
    with tf.gfile.GFile(path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph



class Model(object):
    def __init__(self, map_file, model_file):
        self.char_to_id, self.id_to_tag = load_map(map_file)
        self.graph = load_graph(model_file)
        self.input_x = self.graph.get_tensor_by_name("prefix/char_inputs:0")
        self.lengths = self.graph.get_tensor_by_name("prefix/lengths:0")
        self.dropout = self.graph.get_tensor_by_name("prefix/dropout:0")
        self.logits = self.graph.get_tensor_by_name("prefix/project/logits:0")
        self.trans = self.graph.get_tensor_by_name("prefix/crf_loss/transitions:0")

        self.sess = tf.Session(graph=self.graph)
        self.sess.as_default()
        self.num_class = len(self.id_to_tag)


    def predict(self, text):
        inputs = []
        for w in text:
            if w in self.char_to_id:
                inputs.append(self.char_to_id[w])
            else:
                inputs.append(self.char_to_id["<OOV>"])
        inputs =  np.array(inputs, dtype=np.int32).reshape(1, len(inputs))
        lengths=[len(text)]
        feed_dict = {
            self.input_x: inputs,
            self.lengths: lengths,
            self.dropout: 1.0
        }
        logits, trans = self.sess.run([self.logits, self.trans], feed_dict=feed_dict)
        path = decode(logits, trans, [inputs.shape[1]], self.num_class)
        tags = [self.id_to_tag[p] for p in path]
        return tags
