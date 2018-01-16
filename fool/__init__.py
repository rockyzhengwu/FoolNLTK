#!/usr/bin/env python
# -*-coding:utf-8-*-



import sys
import logging
from collections import defaultdict

from fool import lexical
from fool import dictionary
from  fool import model

LEXICAL_ANALYSER = lexical.LexicalAnalyzer()
_DICTIONARY = dictionary.Dictionary()

__log_console = logging.StreamHandler(sys.stderr)
DEFAULT_LOGGER = logging.getLogger(__name__)
DEFAULT_LOGGER.setLevel(logging.DEBUG)
DEFAULT_LOGGER.addHandler(__log_console)

__all__= ["load_model", "cut", "pos_cut", "ner", "analysis", "load_userdict", "delete_userdict"]

def load_model(map_file, model_file):
    m = model.Model(map_file=map_file, model_file=model_file)
    return m

def _check_input(text, ignore=False):
    if not text:
        return []

    if not isinstance(text, list):
        text = [text]

    null_index = [i for i, t in enumerate(text) if not t]
    if null_index and not ignore:
        raise Exception("null text in input ")

    return text

def ner(text, ignore=False):
    text = _check_input(text, ignore)
    if not text:
        return [[]]
    res = LEXICAL_ANALYSER.ner(text)
    return res


def analysis(text, ignore=False):
    text = _check_input(text, ignore)
    if not text:
        return [[]], [[]]
    res = LEXICAL_ANALYSER.analysis(text)
    return res


def cut(text, ignore=False):

    text = _check_input(text, ignore)

    if not text:
        return [[]]

    text = [t for t in text if t]
    all_words = LEXICAL_ANALYSER.cut(text)
    new_words = []
    if _DICTIONARY.sizes != 0:
        for sent, words in zip(text, all_words):
            words = _mearge_user_words(sent, words)
            new_words.append(words)
    else:
        new_words = all_words
    return new_words


def pos_cut(text):
    words = cut(text)
    pos_labels = LEXICAL_ANALYSER.pos(words)
    word_inf = [list(zip(ws, ps)) for ws, ps in zip(words, pos_labels)]
    return word_inf


def load_userdict(path):
    _DICTIONARY.add_dict(path)


def delete_userdict():
    _DICTIONARY.delete_dict()


def _mearge_user_words(text, seg_results):
    if not _DICTIONARY:
        return seg_results

    matchs = _DICTIONARY.parse_words(text)
    graph = defaultdict(dict)
    text_len = len(text)

    for i in range(text_len):
        graph[i][i + 1] = 1.0

    index = 0

    for w in seg_results:
        w_len = len(w)
        graph[index][index + w_len] = _DICTIONARY.get_weight(w) + w_len
        index += w_len

    for m in matchs:
        graph[m.start][m.end] = _DICTIONARY.get_weight(m.keyword) * len(m.keyword)

    route = {}
    route[text_len] = (0, 0)

    for idx in range(text_len - 1, -1, -1):
        m = [((graph.get(idx).get(k) + route[k][0]), k) for k in graph.get(idx).keys()]
        mm = max(m)
        route[idx] = (mm[0], mm[1])

    index = 0
    path = [index]
    words = []

    while index < text_len:
        ind_y = route[index][1]
        path.append(ind_y)
        words.append(text[index:ind_y])
        index = ind_y

    return words
