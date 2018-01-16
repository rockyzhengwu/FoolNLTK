#!/usr/bin/env python
# -*-coding:utf-8-*-

import fool

text = ["我在北京天安门看你难受香菇,一一千四百二十九", "我在北京晒太阳你在非洲看雪", "千年不变的是什么", "我在北京天安门。"]

print("no dict:", fool.cut(text, ignore=True))
fool.load_userdict("./test_dict.txt")
print("use dict: ", fool.cut(text))
fool.delete_userdict()
print("delete dict:", fool.cut(text))

pos_words =fool.pos_cut(text)
print("pos result", pos_words)

words, ners = fool.analysis(text)
print("ners: ", ners)

ners = fool.ner(text)
print("ners:", ners)