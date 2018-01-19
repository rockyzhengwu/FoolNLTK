#!/usr/bin/env python
# -*-coding:utf-8-*-

import sys
import os
import json

from fool.predictor import Predictor

from zipfile import ZipFile

OOV_STR = "<OOV>"


def _load_map_file(path, char_map_name, id_map_name):
    with ZipFile(path) as myzip:
        with myzip.open('all_map.json') as myfile:
            content = myfile.readline()
            content = content.decode()
            data = json.loads(content)
            return data.get(char_map_name), data.get(id_map_name)


class LexicalAnalyzer(object):
    def __init__(self):
        self.initialized = False
        self.map = None
        self.seg_model = None
        self.pos_model = None
        self.ner_model = None
        self.data_path = os.path.join(sys.prefix, "fool")
        self.map_file_path = os.path.join(self.data_path, "map.zip")


    def _load_model(self, model_namel, word_map_name, tag_name):
        seg_model_path = os.path.join(self.data_path, model_namel)
        char_to_id, id_to_seg = _load_map_file(self.map_file_path, word_map_name, tag_name)
        return Predictor(seg_model_path, char_to_id, id_to_seg)

    def _load_seg_model(self):
        self.seg_model = self._load_model("seg.pb", "char_map", "seg_map")

    def _load_pos_model(self):
        self.pos_model = self._load_model("pos.pb", "word_map", "pos_map")

    def _load_ner_model(self):
        self.ner_model = self._load_model("ner.pb", "char_map", "ner_map")

    def pos(self, seg_words_list):
        if not self.pos_model:
            self._load_pos_model()
        pos_labels = self.pos_model.predict(seg_words_list)
        return pos_labels

    def ner(self, text_list):
        if not self.ner_model:
            self._load_ner_model()

        ner_labels = self.ner_model.predict(text_list)
        all_entitys = []

        for ti, text in enumerate(text_list):
            ens = []
            entity = ""
            i = 0
            ner_label = ner_labels[ti]
            chars = list(text)

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
            all_entitys.append(ens)

        return all_entitys

    def cut(self, text_list):

        if not self.seg_model:
            self._load_seg_model()

        all_labels = self.seg_model.predict(text_list)
        sent_words = []
        for ti, text in enumerate(text_list):
            words = []
            N = len(text)
            seg_labels = all_labels[ti]
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
            sent_words.append(words)
        return sent_words


    def analysis(self, text_list):
        words =  self.cut(text_list)
        pos_labels = self.pos(words)
        ners = self.ner(text_list)
        word_inf = [list(zip(ws, ps)) for ws, ps in zip(words, pos_labels)]
        return word_inf, ners

