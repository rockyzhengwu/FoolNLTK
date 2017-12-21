#!/usr/bin/env python
#-*-coding:utf-8-*-

import fool
from fool.dictionary import Dictionary

def td():
    d = Dictionary()
    d.add_dict("./test_dict.txt")
    matchs = d.parse_words("什么鬼我难受香菇")
    for mat in matchs:
        print(mat.keyword)
        print(mat.start)
        print(mat.end)
    print(d.sizes)


fool.load_userdict("./test_dict.txt")
print(fool._DICTIONARY.sizes)
print(fool._DICTIONARY.weights)

def tcut():
    text = "我在北京天安门"
    words = fool.cut(text)
    print(words)
    fool.delete_userdict()
    print(fool.cut(text))


if __name__ == '__main__':
    td()
    tcut()