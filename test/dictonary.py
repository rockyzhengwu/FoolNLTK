#!/usr/bin/env python
# -*-coding:utf-8-*-

import fool

text = "我在北京天安门看你难受香菇,一一千四百二十九"

print("no dict:", fool.cut(text))
fool.load_userdict("./test_dict.txt")
print("use dict: ", fool.cut(text))
fool.delete_userdict()
print("delete dict:", fool.cut(text))

words, ners = fool.analysis(text)
print("ners: ", ners)
