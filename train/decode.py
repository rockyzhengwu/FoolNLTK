#!/usr/bin/env python
#-*-coding:utf-8-*-

from tensorflow.contrib.crf import viterbi_decode

import numpy as np

# def decode(logits, trans, lengths, num_tags):
#     small = -1000
#
#     start = np.asarray([[small] * num_tags + [0 , small]])
#     end = np.asarray([[small] * num_tags + [small, 0]])
#     path = []
#
#     for logit, length in zip(logits,lengths):
#         logit = logit[ :length]
#         pad = small * np.ones([length, 2])
#         logit = np.concatenate([logit, pad], axis=-1)
#
#         logit = np.concatenate([start, logit, end], axis = 0)
#         viterbi , viterbi_score = viterbi_decode(logit, trans)
#         path.append(np.array(viterbi[1: -1]))
#
#     return path


def vdecode(logits, trans, sequence_lengths, tag_num):
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