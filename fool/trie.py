#!/usr/bin/env python
# -*-coding:utf-8-*-

import queue


class Match(object):
    def __init__(self, start, end, keyword):
        self.start = start
        self.end = end
        self.keyword = keyword

    def __str__(self):
        return str(self.start) + ":" + str(self.end) + "=" + self.keyword


class State(object):
    def __init__(self, word, deepth):
        self.success = {}
        self.failure = None
        self.emits = set()
        self.deepth = deepth

    def add_word(self, word):
        if word in self.success:
            return self.success.get(word)
        else:
            state = State(word, self.deepth + 1)
            self.success[word] = state
        return state

    def add_one_emit(self, keyword):
        self.emits.add(keyword)

    def add_emits(self, emits):
        if not isinstance(emits, set):
            raise Exception("keywords need  a set")
        self.emits = self.emits | emits

    def set_failure(self, state):
        self.failure = state

    def get_transitions(self):
        return self.success.keys()

    def next_state(self, word):
        return self.success.get(word)


class Trie(object):
    def __init__(self, words=[]):

        self.root = State("", 0)
        self.root.set_failure(self.root)
        self.is_create_failure = False
        if words:
            self.create_trie(words)

    def create_trie(self, keyword_list):

        for keyword in keyword_list:
            self.add_keyword(keyword)
        return self

    def add_keyword(self, keyword):
        current_state = self.root
        word_list = list(keyword)

        for word in word_list:
            current_state = current_state.add_word(word)
        current_state.add_one_emit(keyword)

    def create_failure(self):
        root = self.root
        state_queue = queue.Queue()

        for k, v in self.root.success.items():
            state_queue.put(v)
            v.set_failure(root)

        while (not state_queue.empty()):

            current_state = state_queue.get()
            transitions = current_state.get_transitions()

            for word in transitions:
                target_state = current_state.next_state(word)

                state_queue.put(target_state)
                trace_state = current_state.failure

                while (trace_state.next_state(word) is None and trace_state.deepth != 0):
                    trace_state = trace_state.failure

                if trace_state.next_state(word) is not None:
                    target_state.set_failure(trace_state.next_state(word))
                    target_state.add_emits(trace_state.next_state(word).emits)
                else:
                    target_state.set_failure(trace_state)
        self.is_create_failure = True

    def get_state(self, current_state, word):
        new_current_state = current_state.next_state(word)

        while (new_current_state is None and current_state.deepth != 0):
            current_state = current_state.failure
            new_current_state = current_state.next_state(word)

        return new_current_state

    def parse_text(self, text, allow_over_laps=True):
        matchs = []
        if not self.is_create_failure:
            self.create_failure()

        position = 0
        current_state = self.root
        for word in list(text):
            position += 1
            current_state = self.get_state(current_state, word)
            if not current_state:
                current_state = self.root
                continue
            for mw in current_state.emits:
                m = Match(position - len(mw), position, mw)
                matchs.append(m)
        # todo remove over laps
        return matchs


def  create_trie(words):
    return Trie().create_trie(words)
