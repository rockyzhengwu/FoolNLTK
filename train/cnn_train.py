#!/usr/bin/env python
#-*-coding:utf-8-*-


import numpy as np

import data_utils
import word2vec

from collections import OrderedDict
from cnn import CNN
from data_utils import SegBatcher

import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


layers = {'conv1': {'dilation': 1, 'width': 3, 'filters': 300, 'initialization': 'identity', 'take': False}, 'conv2': {'dilation': 2, 'width': 3, 'filters': 300, 'initialization': 'identity', 'take': False}, 'conv3': {'dilation': 1, 'width': 3, 'filters': 300, 'initialization': 'identity', 'take': True}}
print(layers.items())


def init_model_config(vocab_size, tag_size):
     model_config = OrderedDict()
     model_config["char_dim"] = FLAGS.char_dim
     model_config["optimizer"] = FLAGS.optimizer
     model_config['clip'] = FLAGS.clip
     model_config["learning_rate"] = FLAGS.lr
     model_config['word_dropout_keep'] = FLAGS.word_dropout_keep
     model_config["input_dropout_keep"] = FLAGS.input_dropout_keep
     model_config['hidden_dropout_keep'] = FLAGS.hidden_dropout_keep
     model_config["num_chars"] = vocab_size
     model_config["num_tags"] = tag_size
     model_config['decay_rate'] = FLAGS.decay_rate
     model_config['decay_steps'] =FLAGS.decay_steps

     model_config['layers'] = list(layers.items())

     return model_config


def  main(argv):
    print(FLAGS.dev_file)

    word_to_id, tag_to_id, id_to_tag = data_utils.load_map_file(FLAGS.map_file)
    id_to_word = {v:k for k, v in word_to_id.items()}

    num_dict = data_utils.load_size_file(FLAGS.size_file)

    train_num = num_dict['train_num']
    dev_num = num_dict['dev_num']
    test_num = num_dict['test_num']

    model_config = init_model_config(len(word_to_id), len(tag_to_id))
    print(dict(model_config))

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Graph().as_default():

        print("load pre word word2vec ....")
        wv = word2vec.Word2vec()
        embed = wv.load_w2v_array(FLAGS.pre_embedding_file, id_to_word)
        word_embedding = tf.constant(embed, dtype=tf.float32)
        model = CNN(model_config, word_embedding)
        print("batch param =====>  ",FLAGS.batch_size, FLAGS.max_epoch)

        train_batcher = SegBatcher(FLAGS.train_file, FLAGS.batch_size, num_epochs=FLAGS.max_epoch)

        dev_batcher = SegBatcher(FLAGS.dev_file, FLAGS.batch_size, num_epochs=1)
        test_batcher = SegBatcher(FLAGS.test_file, FLAGS.batch_size, num_epochs=1)

        tf.global_variables_initializer()
        sv = tf.train.Supervisor(logdir=FLAGS.out_dir ,
                                 save_model_secs=0,
                                 save_summaries_secs=0)

        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            dev_batches = []
            done = False

            print("load all dev batches to memory")
            while not done:
                try:
                    tags, chars, sent_lens = sess.run(dev_batcher.next_batch_op)
                    dev_batches.append((chars, tags, sent_lens))
                except Exception as e:
                    done = True

            def run_evaluation(dev_batches, report=False):
                preds = []
                true_tags = []
                tmp_x = []
                for x_batch, y_batch, sent_len in dev_batches:
                    feed_dict = {
                        model.input_x: x_batch,
                        model.input_y: y_batch,
                        model.hidden_dropout_keep_prob: FLAGS.hidden_dropout_keep,
                        model.sequence_lengths: sent_len,
                    }

                    step, loss, logits, trans = sess.run(
                        [model.global_step, model.loss, model.block_unflat_scores, model.transition_params], feed_dict)

                    lengths = sent_len.reshape(1, -1)
                    index = 0
                    logits = logits[0]
                    lengths = lengths[0]

                    for score, length in zip(logits, lengths):
                        path, _ = tf.contrib.crf.viterbi_decode(score, trans)
                        preds.append(path[:length])
                        tmp_x.append(x_batch[index][:length])
                        index += 1

                    for y, length in zip(y_batch, lengths):
                        y = y.tolist()
                        true_tags.append(y[: length])

                if report and len(tmp_x) > 5:
                    for j in range(5):
                        sent = [id_to_word.get(i, "<OOV>") for i in tmp_x[j]]
                        print("".join(sent))
                        print("pred:", preds[j])
                        print("true:", true_tags[j])

                preds = np.concatenate(preds, axis=0)
                true_tags = np.concatenate(true_tags, axis=0)
                print(classification_report(true_tags, preds))
                test_acc = accuracy_score(true_tags, preds)
                return test_acc

            def run_test():
                print("start run test ......")
                test_batches = []
                done = False
                print("load all test batches to memory")

                while not done:
                    try:
                        tags, chars, sent_lens = sess.run(test_batcher.next_batch_op)
                        test_batches.append((chars, tags, sent_lens))
                    except:
                        done = True
                test_acc = run_evaluation(test_batches, True)
                print("test accc %f" % (test_acc))

            best_acc = 0.0
            early_stop = False
            loss_history  = []
            for step in range(FLAGS.max_epoch):
                examples = 0

                if sv.should_stop():
                    print("stop ")
                    return

                while examples < train_num:
                    if early_stop:
                        break
                    try:
                        batch = sess.run(train_batcher.next_batch_op)
                    except Exception as e:
                        print("read train batch: ")
                        print(e)
                        break

                    tags, chars, sent_lens = batch
                    examples += tags.shape[0]

                    feed_dict = {
                        model.input_x: chars,
                        model.input_y: tags,
                        model.hidden_dropout_keep_prob: FLAGS.hidden_dropout_keep,
                        model.sequence_lengths: sent_lens,
                    }

                    global_step, batch_loss, _ = sess.run([model.global_step, model.loss, model.train_op], feed_dict)
                    print("%d iteration %d loss: %f" % (step, global_step, batch_loss))
                    if global_step % FLAGS.eval_step ==0:
                        acc = run_evaluation(dev_batches, True)
                        print("%d iteration , %d dev acc: %f " % (step, global_step, acc))
                        if best_acc - acc > 0.01:
                            print("stop training ealy ... best dev acc " % (best_acc))
                            early_stop = True
                            break
                        elif best_acc < acc:
                            best_acc = acc
                            print("%d iteration , %d global step best dev acc: %f " % (step, global_step, best_acc))
                            ckpt = sv.saver.save(sess, FLAGS.out_dir + "model", global_step=global_step)
                            print("save checkpoint:%s"%(ckpt))
                        else:
                            print("%d iteration , %d global step best dev acc: %f " % (step, global_step, best_acc))

                    loss_history.append(batch_loss)
                    examples += FLAGS.batch_size
            ckpt=sv.saver.save(sess, FLAGS.out_dir + "model", global_step=global_step)
            print("save checkpoint: %s"%(ckpt))
            run_test()

        sv.coord.request_stop()
        sv.coord.join(threads)
        sess.close()


if __name__ == "__main__":

    tf.app.flags.DEFINE_string("train_file", "", "train file ")
    tf.app.flags.DEFINE_string("dev_file", "", "dev file")
    tf.app.flags.DEFINE_string("test_file", "", "test file")
    
    tf.app.flags.DEFINE_string("map_file", "", "map file")
    tf.app.flags.DEFINE_string("size_file", "", "size file")
    tf.app.flags.DEFINE_string("pre_embedding_file", "", "pre embedding file")
    tf.app.flags.DEFINE_string("out_dir", "", "model out dir ")

    tf.app.flags.DEFINE_integer("char_dim", 100, "embeding size fo char of word")
    tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer")

    tf.app.flags.DEFINE_float("input_dropout_keep", 1.0, "input drop out")
    tf.app.flags.DEFINE_float("word_dropout_keep", 1.0, "word drop out")
    tf.app.flags.DEFINE_float("hidden_dropout_keep", 0.75, "hidden drop out")
    tf.app.flags.DEFINE_float("lr", 0.001, "learning rate")
    tf.app.flags.DEFINE_integer("clip", 5, "clip")
    tf.app.flags.DEFINE_integer("save_model_secs", 30, "save model secs")

    tf.app.flags.DEFINE_integer("batch_size", 512, "batch size")
    tf.app.flags.DEFINE_integer("max_epoch", 100, "max epoch")
    tf.app.flags.DEFINE_integer("eval_step", 100, "eval step")
    tf.app.flags.DEFINE_string("layers", "", "cnn layers")
    tf.app.flags.DEFINE_integer("decay_steps", 5, "decay step")
    tf.app.flags.DEFINE_float('decay_rate', 0.0001, 'decay rate ')

    FLAGS = tf.flags.FLAGS
    tf.app.run()