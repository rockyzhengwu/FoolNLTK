#-*-coding:utf-8-*-

# !/usr/bin/env python
# -*- coding:utf-8 -*-
# author: wu.zheng midday.me

import collections
import tensorflow as tf
from tensorflow.contrib import predictor
from bert import tokenization
import pickle


VOCAB_FILE = './pretrainmodel/vocab.txt'
LABEL_FILE = './output/label2id.pkl'
EXPORT_PATH = './export_models/1581318324'


def create_int_feature(values):
  f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return f


def convert_single_example(words, label_map, max_seq_length, tokenizer, mode):
  max_seq_length = len(words) + 4
  textlist = words
  tokens = []
  for i, word in enumerate(textlist):
    token = tokenizer.tokenize(word)
    tokens.extend(token)
  if len(tokens) >= max_seq_length - 1:
    tokens = tokens[0:(max_seq_length - 2)]
  ntokens = []
  segment_ids = []
  ntokens.append("[CLS]")
  segment_ids.append(0)
  for i, token in enumerate(tokens):
    ntokens.append(token)
    segment_ids.append(0)
  ntokens.append("[SEP]")
  segment_ids.append(0)
  print(ntokens)
  input_ids = tokenizer.convert_tokens_to_ids(ntokens)
  input_mask = [1] * len(input_ids)
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    ntokens.append("**NULL**")
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  print("input_ids: ", input_ids)
  features = collections.OrderedDict()
  features["input_ids"] = create_int_feature(input_ids)
  features["input_mask"] = create_int_feature(input_mask)
  features["segment_ids"] = create_int_feature(segment_ids)
  return features


class Predictor(object):
  # LABELS= ['O', "B-COMAPNY", "I-COMAPNY", "B-REAL", "I-REAL", "B-AMOUT", "I-AMOUT", "[CLS]", "[SEP]"]
  def __init__(self, export_model_path, vocab_file):
    self.export_model_path = export_model_path
    self.vocab_file = vocab_file
    self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    self.predict_fn = predictor.from_saved_model(self.export_model_path)
    self.label_map = pickle.load(open(LABEL_FILE, 'rb'))
    self.id_to_label = {v: k for k, v in self.label_map.items()}

  def create_example(self, content):
    words = list(content)
    words = [tokenization.convert_to_unicode(w) for w in words]
    features = convert_single_example(words, self.label_map, 256, self.tokenizer, "predict")
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example.SerializeToString()

  def predict(self, conent):
    example_str = self.create_example(content)
    output = self.predict_fn({"inputs": example_str})['output']
    pred_labels = output[0][1:-1]
    predict_labels = [l for l in pred_labels if l != 0]
    print(self.id_to_label)
    print(pred_labels)
    pred_labels = [self.id_to_label[i] for i in predict_labels]
    return pred_labels


if __name__ == "__main__":
  predict_model = Predictor(EXPORT_PATH, VOCAB_FILE)
  content = "河北远大工程咨询有限公司受石家庄市藁城区环境卫生服务站的委托，对石家庄市藁城区2019年新建公厕及垃圾转运站工程进行，并于2019-12-1009:30:00开标、评标，开评标会结束后根据有关法律、法规要求，现对中标候选人进行公员会评审，确定中标候选人为:第一中标候选人:石家庄市藁城区盛安建筑有限公司投标报价：7409295.86第二中标候选人:河北一方建设工程有限公司投标报价：7251181.03第三中标候选人:石家庄卓晟建筑工程有限公司投标报价：709"
  labels = predict_model.predict(content)
  print(labels)
