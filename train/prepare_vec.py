#!/usr/bin/env python
# -*- coding:utf-8 -*-



def load_file(file_name):
    f = open(file_name)
    sent = []
    for line_no, line in enumerate(f):
        line = line.strip("\n")
        if not line:
            yield sent
            sent = []
        else:
            item = line.split("\t")
            sent.append(item)
    f.close()


def papre_char_vec(train_file, dev_file, test_file, out_filename):
    sent_counter = 0
    outf = open(out_filename, 'w')
    for file_name in [train_file, dev_file, test_file]:
        print("papre char vec train data from : %s" % (file_name))
        for sent in load_file(file_name):
            sent_counter += 1
            s = [item[0] for item in sent]
            outf.write(" ".join(s) + "\n")
    print("all sent count %d" % (sent_counter))


def papre_word_vec(train_file, dev_file, test_file, out_filename):
    sent_counter = 0
    outf = open(out_filename, 'w')
    for file_name in [train_file, dev_file, test_file]:

        print("papre word vec train data from : %s" % (file_name))

        for sent in load_file(file_name):
            sent_counter += 1
            word = ""
            words = []
            for item in sent:
                ch = item[0]
                seg_label = item[1]
                if seg_label == "B":
                    word += ch
                elif seg_label == "M":
                    word += ch
                elif seg_label == "S":
                    words.append(ch)
                elif seg_label == "E":
                    word += ch
                    words.append(word)
                    word = ""
                else:
                    raise Exception("%s ignore" % (seg_label))
            outf.write(" ".join(words) + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="train_file path")
    parser.add_argument("--dev_file", required=True, help="dev file path")
    parser.add_argument("--test_file", required=True, help="test file path")
    parser.add_argument("--out_file", required=True, help="out dir for vec path")
    args = parser.parse_args()
    papre_char_vec(args.train_file, args.dev_file, args.test_file, args.out_file)
