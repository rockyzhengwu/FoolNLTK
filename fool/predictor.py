#!/usr/bin/env python
# -*-coding:utf-8-*-



import tensorflow as tf
import numpy as np
from tensorflow.contrib.crf import viterbi_decode


def decode(logits, trans, sequence_lengths, tag_num):
    viterbi_sequences = []
    small = -1000.0
    start = np.asarray([[small] * tag_num + [0]])
    for logit, length in zip(logits, sequence_lengths):
        score = logit[:length]
        pad = small * np.ones([length, 1])
        score = np.concatenate([score, pad], axis=1)
        score = np.concatenate([start, score], axis=0)
        viterbi_seq, viterbi_score = viterbi_decode(score, trans)
        viterbi_sequences.append(viterbi_seq[1:])
    return viterbi_sequences



def list_to_array(data_list, dtype=np.int32):
    array = np.array(data_list, dtype).reshape(1, len(data_list))
    return array


def load_graph(path):
    with tf.gfile.GFile(path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph


class Predictor(object):
    def __init__(self, model_file, char_to_id, id_to_tag):

        self.char_to_id = char_to_id
        self.id_to_tag = {int(k):v for k,v in id_to_tag.items()}
        self.graph = load_graph(model_file)

        self.input_x = self.graph.get_tensor_by_name("prefix/char_inputs:0")
        self.lengths = self.graph.get_tensor_by_name("prefix/lengths:0")
        self.dropout = self.graph.get_tensor_by_name("prefix/dropout:0")
        self.logits = self.graph.get_tensor_by_name("prefix/project/logits:0")
        self.trans = self.graph.get_tensor_by_name("prefix/crf_loss/transitions:0")

        self.sess = tf.Session(graph=self.graph)
        self.sess.as_default()
        self.num_class = len(self.id_to_tag)

    def predict(self, sents):
        inputs = []
        lengths = [len(text) for text in sents]
        max_len = max(lengths)

        for sent in sents:
            sent_ids = [self.char_to_id.get(w) if w in self.char_to_id else self.char_to_id.get("<OOV>") for w in sent]
            padding = [0] * (max_len - len(sent_ids))
            sent_ids += padding
            inputs.append(sent_ids)
        inputs = np.array(inputs, dtype=np.int32)

        feed_dict = {
            self.input_x: inputs,
            self.lengths: lengths,
            self.dropout: 1.0
        }

        logits, trans = self.sess.run([self.logits, self.trans], feed_dict=feed_dict)
        path = decode(logits, trans, lengths, self.num_class)
        labels = [[self.id_to_tag.get(l) for l in p] for p in path]
        return labels
