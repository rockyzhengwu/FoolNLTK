#!/usr/bin/env python
#-*-coding:utf-8-*-

from fool import lexical
from fool import dictionary

LEXICAL_ANALYSER = lexical.LexicalAnalyzer()
_DICTIONARYS = None

def analysis(text):
    res = LEXICAL_ANALYSER.analysis(text)
    return res

def cut(text):
    words, _, _ = LEXICAL_ANALYSER.cut(text)
    return words

def pos_cut(text):
    words, _, _ = LEXICAL_ANALYSER.cut(text)
    pos, _ = LEXICAL_ANALYSER.pos(words)
    return list(zip(words, pos))

def load_userdict(path):
    dt = dictionary.Dictionary(path)
    _DICTIONARYS.append(dt)


def _mearge_user_words(text, seg_results):

    if not _DICTIONARYS:
        return seg_results

    matchs = _DICTIONARYS.parse_words(text)
    graph = {}
    index = 0

    for w in seg_results:
        w_len = len(w)
        if index not in graph:
            graph[index] = []
        graph[index].append(index + w_len)
        index += w_len

    for m in matchs():
        if m.start not in graph:
            graph[m.start] = []
        graph[m.start].append(m.end)

    print(graph)



    # todo 构造此图
    # todo 计算最大权路径
