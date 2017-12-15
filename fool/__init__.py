#!/usr/bin/env python
#-*-coding:utf-8-*-

from fool import lexical
LEXICAL_ANALYSER = lexical.LexicalAnalyzer()

def analysis(text):
    res = LEXICAL_ANALYSER.analylis(text)
    return res

def cut(text):
    words, _, _ = LEXICAL_ANALYSER.cut(text)
    return words

def pos_cut(text):
    words, _, _ = LEXICAL_ANALYSER.cut(text)
    pos, _ = LEXICAL_ANALYSER.pos(words)
    return list(zip(words, pos))