#!/usr/bin/env python
# -*-coding:utf-8-*-



import sys
import time
import logging
from collections import defaultdict

from fool import lexical
from fool import dictionary

LEXICAL_ANALYSER = lexical.LexicalAnalyzer()
_DICTIONARY = dictionary.Dictionary()

__log_console = logging.StreamHandler(sys.stderr)
DEFAULT_LOGGER = logging.getLogger(__name__)
DEFAULT_LOGGER.setLevel(logging.DEBUG)
DEFAULT_LOGGER.addHandler(__log_console)

def _check_model():
    if not LEXICAL_ANALYSER.initialized:
        DEFAULT_LOGGER.debug("starting load model ")
        start = time.time()
        LEXICAL_ANALYSER.load_model()
        DEFAULT_LOGGER.debug("loaded model cost : %fs" % (time.time() - start))

def analysis(text):
    _check_model()
    res = LEXICAL_ANALYSER.analysis(text)
    return res


def cut(text):
    _check_model()
    if not text:
        return []
    words, _, _ = LEXICAL_ANALYSER.cut(text)

    if _DICTIONARY.sizes != 0:
        words = _mearge_user_words(text, words)
    return words


def pos_cut(text):
    words = cut(text)
    pos, _ = LEXICAL_ANALYSER.pos(words)
    return list(zip(words, pos))


def load_userdict(path):
    _DICTIONARY.add_dict(path)


def delete_userdict():
    _DICTIONARY.delete_dict()


def _mearge_user_words(text, seg_results):
    # todo 根据词权重合并，实现全局最优

    if not _DICTIONARY:
        return seg_results

    matchs = _DICTIONARY.parse_words(text)
    graph = defaultdict(dict)
    index = 0

    for w in seg_results:
        w_len = len(w)
        graph[index][index + w_len] = _DICTIONARY.get_weight(w) + w_len
        index += w_len

    for m in matchs:
        graph[m.start][m.end] = _DICTIONARY.get_weight(m.keyword) + len(m.keyword)

    start_cursor = 0
    text_len = len(text)
    paths = []

    while start_cursor < text_len:
        if start_cursor not in graph:
            paths.append((start_cursor, start_cursor + 1))
            start_cursor += 1
            continue
        trans = graph.get(start_cursor)
        dist = max(trans.items(), key=lambda x: x[1])[0]
        paths.append((start_cursor, dist))
        start_cursor = dist

    words = [text[p[0]:p[1]] for p in paths]
    return words
