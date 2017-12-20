#!/usr/bin/env python
#-*-coding:utf-8-*-

import trie

class Dictionary():
    def __init__(self):
        self.trie = trie.Trie()
        self.weights = {}

    def add_dict(self, path):
        words = []
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip("\n").strip()
                if not line:
                    continue
                line = line.split("\t")
                word = line[0]

                if len(line)==1:
                    weight = 5.0
                else:
                    weight = float(line[1])
                weight = float(weight)
                self.weights[word] = weight
                words.append(word)


    def parse_words(self, text):
        matchs = self.trie.parse_text(text)
        return matchs

    def get_weight(self, word):
        self.weights.get(word, 0)




if __name__ == '__main__':
    d = Dictionary("../data/test_dict.txt")
    words = d.parse_words("什么鬼我难受香菇")
    print(words)

