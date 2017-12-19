#!/usr/bin/env python
# -*-coding:utf-8-*-



import tensorflow as tf
import numpy as np
from tensorflow.contrib.crf import viterbi_decode

SEG_DICT = {"B": 0, "M": 1, "E": 2, "S": 3}

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
        viterbi_sequences += [viterbi_seq]
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
    def __init__(self, model_path, num_class):
        self.graph = load_graph(model_path)
        self.input_x = self.graph.get_tensor_by_name("prefix/chatInputs:0")
        self.dropout = self.graph.get_tensor_by_name("prefix/dropout:0")
        self.logits = self.graph.get_tensor_by_name("prefix/project/logits:0")
        self.trans = self.graph.get_tensor_by_name("prefix/crf_loss/transitions:0")

        self.sess = tf.Session(graph=self.graph)
        self.sess.as_default()
        self.num_class = num_class

    def predict(self, inputs):
        inputs = list_to_array(inputs)

        feed_dict = {
            self.input_x: inputs,
            self.dropout: 1.0
        }
        logits, trans = self.sess.run([self.logits, self.trans], feed_dict=feed_dict)
        path = decode(logits, trans, [inputs.shape[1]], self.num_class)
        path = path[0][1:]
        return path

class NPredictor(object):
    def __init__(self, model_path, num_class, is_pos=False):
        self.graph = load_graph(model_path)
        if is_pos:
            self.pos_in = self.graph.get_tensor_by_name("prefix/pos:0")
        self.is_pos = is_pos

        self.input_x = self.graph.get_tensor_by_name("prefix/chatInputs:0")
        self.segs = self.graph.get_tensor_by_name("prefix/segs:0")
        self.dropout = self.graph.get_tensor_by_name("prefix/dropout:0")
        self.logits = self.graph.get_tensor_by_name("prefix/project/logits:0")
        self.trans = self.graph.get_tensor_by_name("prefix/crf_loss/transitions:0")

        self.sess = tf.Session(graph=self.graph)
        self.sess.as_default()
        self.num_class = num_class

    def predict(self, char_inputs, segs, pos=[]):

        seg_inputs = [SEG_DICT.get(s) for s in segs]

        chars = list_to_array(char_inputs)
        segs = list_to_array(seg_inputs)

        if not self.is_pos:
            feed_dict = {
                self.input_x: chars,
                self.segs: segs,
                self.dropout: 1.0
            }
        else:
            poss = list_to_array(pos)
            feed_dict = {
                self.input_x: chars,
                self.segs: segs,
                self.pos_in: poss,
                self.dropout: 1.0
            }
        logits, trans = self.sess.run([self.logits, self.trans], feed_dict=feed_dict)
        path = decode(logits, trans, [chars.shape[1]], self.num_class)
        path = path[0][1:]
        return path
