#!/usr/bin/env python
# -*-coding:utf-8-*-


from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
import tf_utils


class CNN(object):

    def __init__(self, config, embeddings=None):

        self.num_classes = config["num_tags"]
        self.nonlinearity = "relu"
        self.projection = True
        self.embedding_size = config['char_dim']
        self.repeats = 1
        self.viterbi = True
        self.share_repeats = 1
        self.vocab_size = config['num_chars']
        self.layers_map = config['layers']
        self.learning_rate = config['learning_rate']
        self.decay_steps = config['decay_steps']
        self.decay_rate = config['decay_rate']
        self.optimizer = config['optimizer']

        # word embedding input
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")

        # labels
        self.input_y = tf.placeholder(tf.int32, [None, None], name="input_y")

        # dims
        self.batch_size = tf.shape(self.input_x)[0]
        self.max_seq_len = tf.shape(self.input_x)[1]

        self.global_step = tf.Variable(0, trainable=False)

        # sequence lengths
        self.sequence_lengths = tf.placeholder(tf.int32, [None, None], name="sequence_lengths")

        # dropout and l2 penalties
        self.hidden_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="hidden_dropout_keep_prob")
        self.input_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="input_dropout_keep_prob")
        self.middle_dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="middle_dropout_keep_prob")

        self.training = tf.placeholder_with_default(False, [], name="training")

        self.l2_penalty = tf.placeholder_with_default(0.0, [], name="l2_penalty")
        self.drop_penalty = tf.placeholder_with_default(0.0, [], name="drop_penalty")

        self.l2_loss = tf.constant(0.0)

        self.ones = tf.ones([self.batch_size, self.max_seq_len, self.num_classes])

        self.transition_params = tf.get_variable("transitions", [self.num_classes, self.num_classes])

        word_embeddings_shape = (self.vocab_size-1, self.embedding_size)

        self.w_e = tf_utils.initialize_embeddings(word_embeddings_shape, name="w_e", pretrained=embeddings, old=False)

        self.block_unflat_scores, self.hidden_layer = self.forward(self.input_x, self.max_seq_len,
                                          self.hidden_dropout_keep_prob,
                                          self.input_dropout_keep_prob,
                                          self.middle_dropout_keep_prob, reuse=False)



        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            self.loss = tf.constant(0.0)
            self.block_unflat_no_dropout_scores, _ = self.forward(self.input_x, self.max_seq_len, 1.0, 1.0, 1.0)
            labels = tf.cast(self.input_y, 'int32')
            self.unflat_scores = self.block_unflat_scores[-1]
            self.unflat_no_dropout_scores = self.block_unflat_no_dropout_scores[-1]
            self.loss = self.compute_loss(self.unflat_scores, self.unflat_no_dropout_scores, labels)

        with tf.name_scope("predictions"):
            self.predictions = tf.identity(self.unflat_scores, name="predict")

        self.train_op = self.train()


    def compute_loss(self, scores, scores_no_dropout, labels):

        loss = tf.constant(0.0)

        if self.viterbi:
            zero_elements = tf.equal(self.sequence_lengths, tf.zeros_like(self.sequence_lengths))
            count_zeros_per_row = tf.reduce_sum(tf.to_int32(zero_elements), axis=1)
            flat_sequence_lengths = tf.add(tf.reduce_sum(self.sequence_lengths, 1),
                                           tf.scalar_mul(2, count_zeros_per_row))
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(scores, labels, flat_sequence_lengths,
                                                                                  transition_params=self.transition_params)
            loss += tf.reduce_mean(-log_likelihood)

        loss += self.l2_penalty * self.l2_loss
        drop_loss = tf.nn.l2_loss(tf.subtract(scores, scores_no_dropout))
        loss += self.drop_penalty * drop_loss

        return loss

    def forward(self, input_x1, max_seq_len, hidden_dropout_keep_prob, input_dropout_keep_prob,
                middle_dropout_keep_prob, reuse=True):

        block_unflat_scores = []

        with tf.variable_scope("forward", reuse=reuse):
            word_embeddings = tf.nn.embedding_lookup(self.w_e, input_x1)

            input_list = [word_embeddings]
            input_size = self.embedding_size

            initial_filter_width = self.layers_map[0][1]['width']
            initial_num_filters = self.layers_map[0][1]['filters']

            filter_shape = [1, initial_filter_width, input_size, initial_num_filters]

            initial_layer_name = "conv0"


            if not reuse:
                print(input_list)
                print("Adding initial layer %s: width: %d; filters: %d" % (
                    initial_layer_name, initial_filter_width, initial_num_filters))

            input_feats = tf.concat(axis=2, values=input_list)
            input_feats_expanded = tf.expand_dims(input_feats, 1)
            input_feats_expanded_drop = tf.nn.dropout(input_feats_expanded, input_dropout_keep_prob)
            print("input feats expanded drop", input_feats_expanded_drop.get_shape())

            # first projection of embeddings
            w = tf_utils.initialize_weights(filter_shape, initial_layer_name + "_w", init_type='xavier', gain='relu')
            b = tf.get_variable(initial_layer_name + "_b", initializer=tf.constant(0.01, shape=[initial_num_filters]))
            conv0 = tf.nn.conv2d(input_feats_expanded_drop, w, strides=[1, 1, 1, 1], padding="SAME", name=initial_layer_name)
            h0 = tf_utils.apply_nonlinearity(tf.nn.bias_add(conv0, b), 'relu')

            initial_inputs = [h0]
            last_dims = initial_num_filters

            # Stacked atrous convolutions
            last_output = tf.concat(axis=3, values=initial_inputs)

            for block in range(self.repeats):
                hidden_outputs = []
                total_output_width = 0
                reuse_block = (block != 0 and self.share_repeats) or reuse
                block_name_suff = "" if self.share_repeats else str(block)
                inner_last_dims = last_dims
                inner_last_output = last_output
                with tf.variable_scope("block" + block_name_suff, reuse=reuse_block):
                    for layer_name, layer in self.layers_map:
                        dilation = layer['dilation']
                        filter_width = layer['width']
                        num_filters = layer['filters']
                        initialization = layer['initialization']
                        take_layer = layer['take']

                        if not reuse:
                            print("Adding layer %s: dilation: %d; width: %d; filters: %d; take: %r" % (
                            layer_name, dilation, filter_width, num_filters, take_layer))
                        with tf.name_scope("atrous-conv-%s" % layer_name):
                            # [filter_height, filter_width, in_channels, out_channels]
                            filter_shape = [1, filter_width, inner_last_dims, num_filters]
                            w = tf_utils.initialize_weights(filter_shape, layer_name + "_w", init_type=initialization, gain=self.nonlinearity, divisor=self.num_classes)
                            b = tf.get_variable(layer_name + "_b", initializer=tf.constant(0.0 if initialization == "identity" or initialization == "varscale" else 0.001, shape=[num_filters]))
                            conv = tf.nn.atrous_conv2d(inner_last_output, w, rate=dilation, padding="SAME", name=layer_name)
                            conv_b = tf.nn.bias_add(conv, b)
                            h = tf_utils.apply_nonlinearity(conv_b, self.nonlinearity)

                            if take_layer:
                                hidden_outputs.append(h)
                                total_output_width += num_filters
                            inner_last_dims = num_filters
                            inner_last_output = h

                    h_concat = tf.concat(axis=3, values=hidden_outputs)

                    last_output = tf.nn.dropout(h_concat, middle_dropout_keep_prob)
                    last_dims = total_output_width

                    h_concat_squeeze = tf.squeeze(h_concat, [1])
                    h_concat_flat = tf.reshape(h_concat_squeeze, [-1, total_output_width])
                    print(h_concat_flat)

                    # Add dropout
                    with tf.name_scope("hidden_dropout"):
                        h_drop = tf.nn.dropout(h_concat_flat, hidden_dropout_keep_prob)

                    def do_projection():
                        # Project raw outputs down
                        with tf.name_scope("projection"):
                            projection_width = int(total_output_width/(2*len(hidden_outputs)))
                            w_p = tf_utils.initialize_weights([total_output_width, projection_width], "w_p", init_type="xavier")
                            b_p = tf.get_variable("b_p", initializer=tf.constant(0.01, shape=[projection_width]))
                            projected = tf.nn.xw_plus_b(h_drop, w_p, b_p, name="projected")
                            projected_nonlinearity = tf_utils.apply_nonlinearity(projected, self.nonlinearity)
                        return projected_nonlinearity, projection_width

                    # only use projection if we wanted to, and only apply middle dropout here if projection
                    input_to_pred, proj_width = do_projection() if self.projection else (h_drop, total_output_width)
                    input_to_pred_drop = tf.nn.dropout(input_to_pred, middle_dropout_keep_prob) if self.projection else input_to_pred

                    # Final (unnormalized) scores and predictions
                    with tf.name_scope("output"+block_name_suff):
                        w_o = tf_utils.initialize_weights([proj_width, self.num_classes], "w_o", init_type="xavier")
                        b_o = tf.get_variable("b_o", initializer=tf.constant(0.01, shape=[self.num_classes]))
                        self.l2_loss += tf.nn.l2_loss(w_o)
                        self.l2_loss += tf.nn.l2_loss(b_o)
                        scores = tf.nn.xw_plus_b(input_to_pred_drop, w_o, b_o, name="scores")
                        unflat_scores = tf.reshape(scores, tf.stack([self.batch_size, max_seq_len, self.num_classes]))
                        block_unflat_scores.append(unflat_scores)

        return block_unflat_scores, h_concat_squeeze


    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                   self.global_step,
                                                   self.decay_steps,
                                                   self.decay_rate,
                                                   staircase=True)
        # learning_rate = self.learning_rate
        train_op = tf.contrib.layers.optimize_loss(self.loss,
                                                        global_step=self.global_step,
                                                        learning_rate=learning_rate,
                                                        optimizer=self.optimizer,
                                                        )
        return train_op