#!/usr/bin/env python
# -*-coding:utf-8-*-

import sys
import os
import pickle

from fool.predictor import Predictor, NPredictor

OOV_STR = "<OOV>"


class DataMap(object):
    def __init__(self, path):
        self.char_to_id = {}
        self.word_to_id = {}
        self.id_to_seg = {}
        self.id_to_pos = {}
        self.id_to_ner = {}
        self._load(path)
        self.num_seg = len(self.id_to_seg)
        self.num_pos = len(self.id_to_pos)
        self.num_ner = len(self.id_to_ner)

    def _load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.char_to_id = data.get("char_map")
            self.word_to_id = data.get("word_map")
            self.id_to_seg = data.get("seg_map")
            self.id_to_pos = data.get("pos_map")
            self.id_to_ner = data.get("ner_map")

    def map_id_label(self, ids, dict_name):
        if dict_name == "seg":
            labels = [self.id_to_seg.get(i) for i in ids]
        elif dict_name == "pos":
            labels = [self.id_to_pos.get(i) for i in ids]
        elif dict_name == "ner":
            labels = [self.id_to_ner.get(i) for i in ids]
        else:
            raise Exception("dict_name {} not error".format(dict_name))
        return labels

    def map_char(self, chars):
        vec = []
        for c in chars:
            if c in self.char_to_id:
                vec.append(self.char_to_id[c])
            else:
                vec.append(self.char_to_id[OOV_STR])
        return vec

    def map_word(self, words):
        vec = []
        for w in words:
            if w in self.word_to_id:
                vec.append(self.word_to_id[w])
            else:
                vec.append(self.word_to_id[OOV_STR])
        return vec


class LexicalAnalyzer(object):
    def __init__(self):
        self.initialized = False
        self.map = None
        self.seg_model = None
        self.pos_model = None
        self.ner_model = None


    def load_model(self):
        data_path = os.path.join(sys.prefix, "fool")
        self.map = DataMap(os.path.join(data_path, "maps.pkl"))
        self.seg_model = Predictor(os.path.join(data_path, "seg.pb"), self.map.num_seg)
        self.pos_model = Predictor(os.path.join(data_path, "pos.pb"), self.map.num_pos)
        self.ner_model = NPredictor(os.path.join(data_path, "ner_pos.pb"), self.map.num_ner, True)
        self.initialized = True


    def pos(self, words):
        if not words:
            return [], []
        word_vec = self.map.map_word(words)
        pos_path = self.pos_model.predict(word_vec)
        pos = self.map.map_id_label(pos_path, "pos")
        return pos, pos_path

    def ner(self, chars, char_ids, seg_ids, pos_ids=[]):

        ner_path = self.ner_model.predict(char_ids, seg_ids, pos_ids)
        ner_label = self.map.map_id_label(ner_path, "ner")
        ens = []
        entity = ""
        i = 0

        for label, word in zip(ner_label, chars):
            i += 1
            if label == "O":
                continue
            lt = label.split("_")[1]
            lb = label.split("_")[0]
            if lb == "S":
                ens.append((i, i + 1, lt, word))
            elif lb == "B":
                entity = ""
                entity += word
            elif lb == "M":
                entity += word
            elif lb == "E":
                entity += word
                ens.append((i - len(entity), i + 1, lt, entity))
                entity = ""
        if entity:
            ens.append((i - len(entity), i + 1, lt, entity))
        return ens

    def cut(self, text):
        if not text:
            return [], [], []
        input_chars = self.map.map_char(list(text))
        seg_path = self.seg_model.predict(input_chars)
        seg_labels = self.map.map_id_label(seg_path, "seg")

        N = len(seg_labels)
        words = []
        tmp_word = ""
        for i in range(N):
            label = seg_labels[i]
            w = text[i]
            if label == "B":
                tmp_word += w
            elif label == "M":
                tmp_word += w
            elif label == "E":
                tmp_word += w
                words.append(tmp_word)
                tmp_word = ""
            else:
                tmp_word = ""
                words.append(w)

        if tmp_word:
            words.append(tmp_word)
        return words, input_chars, seg_labels

    def analysis(self, text):
        if not text:
            return [], []
        words, input_chars,  seg_labels = self.cut(text)
        ps, pos_path = self.pos(words)
        ner_pos_path = []
        for w, p in zip(words, pos_path):
            ner_pos_path.extend([p] * len(w))
        entitys = self.ner(list(text), input_chars, seg_labels, ner_pos_path)
        return list(zip(words, ps)), entitys
