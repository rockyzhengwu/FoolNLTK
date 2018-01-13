#!/usr/bin/env python
# -*-coding:utf-8-*-


from collections import OrderedDict

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import data_utils
from bi_lstm import BiLSTM
import word2vec
from data_utils import SegBatcher


def init_mode_config(vocab_size, tag_size):
    model_config = OrderedDict()

    model_config["char_dim"] = FLAGS.char_dim
    model_config["lstm_dim"] = FLAGS.lstm_dim
    model_config["optimizer"] = FLAGS.optimizer
    model_config['clip'] = FLAGS.clip
    model_config["lr"] = FLAGS.lr
    model_config['dropout'] = FLAGS.dropout
    model_config["input_dropout_keep"] = FLAGS.input_dropout_keep
    model_config["num_chars"] = vocab_size
    model_config["num_tags"] = tag_size

    return model_config


def main(argv):
    # todo create map file
    word_to_id, tag_to_id, id_to_tag = data_utils.load_map_file(FLAGS.map_file)
    id_to_word = {v: k for k, v in word_to_id.items()}

    num_dict = data_utils.load_size_file(FLAGS.size_file)
    train_num = num_dict["train_num"]
    dev_num = num_dict["dev_num"]
    test_num = num_dict['test_num']

    model_config = init_mode_config(len(word_to_id), len(tag_to_id))
    print(model_config)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Graph().as_default():

        print("load pre word2vec ...")
        wv = word2vec.Word2vec()
        embed = wv.load_w2v_array(FLAGS.pre_embedding_file, id_to_word)

        word_embedding = tf.constant(embed, dtype=tf.float32)
        model = BiLSTM(model_config, word_embedding)
        train_batcher = SegBatcher(FLAGS.train_file, FLAGS.batch_size, num_epochs=FLAGS.max_epoch)
        dev_batcher = SegBatcher(FLAGS.dev_file, FLAGS.batch_size, num_epochs=1)
        test_batcher = SegBatcher(FLAGS.test_file, FLAGS.batch_size, num_epochs=1)

        tf.global_variables_initializer()
        sv = tf.train.Supervisor(logdir=FLAGS.out_dir, save_model_secs=FLAGS.save_model_secs, )

        with sv.managed_session() as sess:
            sess.as_default()
            threads = tf.train.start_queue_runners(sess=sess)
            loss = []


            def run_evaluation(dev_batches, report=False):
                """
                Evaluates model on a dev set
                """
                preds = []
                true_tags = []
                tmp_x = []
                for x_batch, y_batch, sent_len in dev_batches:
                    feed_dict = {
                        model.char_inputs: x_batch,
                        model.targets: y_batch,
                        model.lengths: sent_len.reshape(-1, ),
                        model.dropout: 1.0
                    }

                    step, loss, logits, lengths, trans = sess.run(
                        [model.global_step, model.loss, model.logits, model.lengths, model.trans], feed_dict)

                    index = 0
                    small = -1000.0
                    start = np.asarray([[small] * model_config["num_tags"] + [0]])

                    for score, length in zip(logits, lengths):
                        score = score[:length]
                        pad = small * np.ones([length, 1])
                        logit = np.concatenate([score, pad], axis=1)
                        logit = np.concatenate([start, logit], axis=0)
                        path, _ = tf.contrib.crf.viterbi_decode(logit, trans)
                        preds.append(path[1:])
                        tmp_x.append(x_batch[index][:length])
                        index += 1

                    for y, length in zip(y_batch, lengths):
                        y = y.tolist()
                        true_tags.append(y[: length])

                if FLAGS.debug and len(tmp_x) > 5:
                    print(tag_to_id)

                    for j in range(5):
                        sent = [id_to_word.get(i, "<OOV>") for i in tmp_x[j]]
                        print("".join(sent))
                        print("pred:", preds[j])
                        print("true:", true_tags[j])

                preds = np.concatenate(preds, axis=0)
                true_tags = np.concatenate(true_tags, axis=0)

                if report:
                    print(classification_report(true_tags, preds))

                acc = accuracy_score(true_tags, preds)
                return acc

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
            dev_batches = []
            done = False
            print("load all dev batches to memory")

            while not done:
                try:
                    tags, chars, sent_lens = sess.run(dev_batcher.next_batch_op)
                    dev_batches.append((chars, tags, sent_lens))
                except:
                    done = True

            print("start training ...")
            early_stop = False
            for step in range(FLAGS.max_epoch):
                if sv.should_stop():
                    run_test()
                    break
                examples = 0

                while examples < train_num:
                    if early_stop:
                        break
                    try:
                        batch = sess.run(train_batcher.next_batch_op)
                    except Exception as e:
                        break

                    tags, chars, sent_lens = batch
                    feed_dict = {
                        model.char_inputs: chars,
                        model.targets: tags,
                        model.dropout: FLAGS.dropout,
                        model.lengths: sent_lens.reshape(-1, ),
                    }
                    global_step, batch_loss, _ = sess.run([model.global_step, model.loss, model.train_op], feed_dict)

                    print("%d iteration %d loss: %f" % (step, global_step, batch_loss))
                    if global_step % FLAGS.eval_step == 0:
                        print("evaluation .......")
                        acc = run_evaluation(dev_batches)

                        print("%d iteration , %d dev acc: %f " % (step, global_step, acc))

                        if best_acc - acc > 0.01:
                            print("stop training ealy ... best dev acc " % (best_acc))
                            early_stop = True
                            break

                        elif best_acc < acc:
                            best_acc = acc
                            sv.saver.save(sess, FLAGS.out_dir + "model", global_step=global_step)
                            print("%d iteration , %d global step best dev acc: %f " % (step, global_step, best_acc))

                    loss.append(batch_loss)
                    examples += FLAGS.batch_size

            sv.saver.save(sess, FLAGS.out_dir + "model", global_step=global_step)
            run_test()
        sv.coord.request_stop()
        sv.coord.join(threads)
        sess.close()


if __name__ == "__main__":
    tf.app.flags.DEFINE_string("train_file", "", "path of train recoard path")
    tf.app.flags.DEFINE_string("dev_file", "", "path of dev recoard path")
    tf.app.flags.DEFINE_string("test_file", "", "path of dev recoard path")

    tf.app.flags.DEFINE_string("pre_embedding_file", "", "vec of char or map file path")
    tf.app.flags.DEFINE_string("map_file", "", "map file ")
    tf.app.flags.DEFINE_string("out_dir", "", "log path of the supervisor")
    tf.app.flags.DEFINE_string("size_file", "", "size file")

    tf.app.flags.DEFINE_integer("max_epoch", 20, "max epoch")
    tf.app.flags.DEFINE_integer("batch_size", 32, "batch size")

    tf.app.flags.DEFINE_float("input_dropout_keep", 1.0, "input drop out ")
    tf.app.flags.DEFINE_float("dropout", 0.5, "dropout")

    tf.app.flags.DEFINE_integer("eval_step", 10, "evaluation step size")

    tf.app.flags.DEFINE_integer("char_dim", 100, "the embedding size of char or word")
    tf.app.flags.DEFINE_string("optimizer", "adam", "optimizer ")
    tf.app.flags.DEFINE_integer("clip", 5, "clip  ")
    tf.app.flags.DEFINE_integer("lstm_dim", 100, "lstm dim")
    tf.app.flags.DEFINE_float("lr", 0.001, "learning rate")
    tf.app.flags.DEFINE_integer("save_model_secs", 30, "save model every second")

    tf.app.flags.DEFINE_boolean("debug", True, "if debug ")

    FLAGS = tf.flags.FLAGS
    tf.app.run()
