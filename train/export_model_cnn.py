#!/usr/bin/env python
#-*-coding:utf-8-*-


import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
import numpy as np

def decode(logits, trans, sequence_lengths, tag_num):
    viterbi_sequences = []
    for logit, length in zip(logits, sequence_lengths):
        viterbi_seq, viterbi_score = viterbi_decode(logit, trans)
        viterbi_sequences += viterbi_seq[:length]
    return viterbi_sequences


def save_to_binary(checkpoints_path, out_model_path, out_put_names):
    checkpoint_dir = checkpoints_path
    graph = tf.Graph()
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    print(checkpoint_file)

    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )

        sess = tf.Session(config=session_conf)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names= out_put_names
        )

        with tf.gfile.FastGFile(out_model_path, mode='wb') as f:
            f.write(output_graph_def.SerializeToString())


import pickle
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

class Predictor(object):
    def __init__(self, map_path, model_path):
        self.char_to_id, self.id_to_tag = load_map(map_path)
        self.graph = load_graph(model_path)
        # for op in self.graph.get_operations():
        #     print(op.name)

        self.input_x = self.graph.get_tensor_by_name("prefix/input_x:0")
        self.lengths = self.graph.get_tensor_by_name("prefix/sequence_lengths:0")
        self.trans= self.graph.get_tensor_by_name("prefix/transitions:0")
        self.logits = self.graph.get_tensor_by_name("prefix/predictions/predict:0")

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
        lengths = np.array([len(text)]).reshape(1, -1)

        feed_dict = {
            self.input_x: inputs,
            self.lengths: lengths
        }

        logits, trans = self.sess.run([self.logits, self.trans], feed_dict=feed_dict)

        path = decode(logits, trans, [inputs.shape[1]], self.num_class)
        tags = [self.id_to_tag[p] for p in path]
        return tags


if __name__ == '__main__':

    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="checkpoint dir")
    parser.add_argument("--out_dir", required=True, help="out dir ")
    args = parser.parse_args()
    print("export model from: %s, save to :%s"%(args.checkpoint_dir, args.out_dir) )
    out_names  = ['predictions/predict', "transitions", "sequence_lengths"]
    save_to_binary(args.checkpoint_dir, os.path.join(args.out_dir, "modle.pb"), out_names)

    # import time
    # model_path = "/home/wuzheng/dl/FoolNLTK-train/train/results/seg_cnn/modle.pb"
    # map_path = "/home/wuzheng/dl/FoolNLTK-train/train/datasets/seg/maps.pkl"
    # predictor = Predictor(map_path=map_path, model_path=model_path)
    # start = time.time()
    # predictor.predict("北京欢迎你"* 200)
    # print(time.time() - start)