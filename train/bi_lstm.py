#!/usr/bin/env python
# -*-coding:utf-8-*-


import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.layers.python.layers import initializers


class BiLSTM(object):
    def __init__(self, config, embeddings):

        self.config = config

        self.lstm_dim = config["lstm_dim"]
        self.num_chars = config["num_chars"]
        self.num_tags = config["num_tags"]
        self.char_dim = config["char_dim"]
        self.lr = config["lr"]


        self.char_embeding = tf.get_variable(name="char_embeding", initializer=embeddings)

        self.global_step = tf.Variable(0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="char_inputs")
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="targets")
        self.dropout = tf.placeholder(dtype=tf.float32, name="dropout")
        self.lengths = tf.placeholder(dtype=tf.int32, shape=[None, ], name="lengths")


        # self.middle_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="middle_dropout_keep_prob")
        # self.hidden_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="hidden_dropout_keep_prob")

        self.input_dropout_keep_prob = tf.placeholder_with_default(config["input_dropout_keep"], [], name="input_dropout_keep_prob")

        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        # forward
        embedding = self.embedding_layer(self.char_inputs)
        lstm_inputs = tf.nn.dropout(embedding, self.input_dropout_keep_prob)

        ## bi-directional lstm layer
        lstm_outputs = self.bilstm_layer(lstm_inputs)
        ## logits for tags
        self.project_layer(lstm_outputs)
        ## loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)


        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v] for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)


    def embedding_layer(self, char_inputs):
        with tf.variable_scope("char_embedding"), tf.device('/cpu:0'):
            embed = tf.nn.embedding_lookup(self.char_embeding, char_inputs)
        return embed


    def bilstm_layer(self, lstm_inputs, name=None):
        with tf.variable_scope("char_bilstm" if not name else name):
            lstm_fw_cell = rnn.BasicLSTMCell(self.lstm_dim, state_is_tuple=True)
            lstm_bw_cell = rnn.BasicLSTMCell(self.lstm_dim, state_is_tuple=True)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, lstm_inputs, dtype=tf.float32, sequence_length=self.lengths)
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        """
        # 尝试 todo 直接映射lstm_outputs 到 num_tags

        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                w_tanh = tf.get_variable("w_tanh", shape=[self.lstm_dim * 2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer, regularizer=tf.contrib.layers.l2_regularizer(0.001))

                b_tanh = tf.get_variable("b_tanh", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, w_tanh, b_tanh))

                drop_hidden = tf.nn.dropout(hidden, self.dropout)

            # output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim * 2])
            # drop_hidden = tf.nn.dropout(output, 1-self.dropout)

            # project to score of tags
            with tf.variable_scope("output"):
                w_out = tf.get_variable("w_out", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer, regularizer=tf.contrib.layers.l2_regularizer(0.001))

                b_out = tf.get_variable("b_out", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer() )
                pred = tf.nn.xw_plus_b(drop_hidden, w_out, b_out, name="pred")
            self.logits = tf.reshape(pred, [-1, self.num_steps, self.num_tags], name="logits")


    def loss_layer(self, project_logits, lengths, name=None):

        with tf.variable_scope("crf_loss" if not name else name):
            small = -1000.0
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])],
                axis=-1)

            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)

            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths + 1)

            return tf.reduce_mean(-log_likelihood)
