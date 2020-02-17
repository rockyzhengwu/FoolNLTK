# FoolNLTK-train
FoolNLTK training process

增加了利用 bert 训练实体模型的代码，可以导出 pb 文件，并用 python 加载实现线上部署
模型没使用 crf ，而是直接用交叉熵作为损失，效果并没有太多损失

1. 模型训练
data_dir 存放训练数据格式如 datasets/demo 下。下载与训练的模型,我这里是将下载的模型软链接到 pretrainmodel 下 

```shell script

python ./train_bert_ner.py --data_dir=data/bid_train_data \
  --bert_config_file=./pretrainmodel/bert_config.json \
  --init_checkpoint=./pretrainmodel/bert_model.ckpt \
  --vocab_file=./pretrainmodel/vocab.txt \
  --output_dir=./output/all_bid_result_dir/ --do_train

```

2. 模型导出
 predict 同时指定 do_export 就能导出 pb 格式的模型，用于部署
```shell script
python ./train_bert_ner.py --data_dir=data/bid_train_data \
  --bert_config_file=./pretrainmodel/bert_config.json \
  --init_checkpoint=./pretrainmodel/bert_model.ckpt \
  --vocab_file=vocab.txt \
  --output_dir=./output/all_bid_result_dir/ --do_predict --do_export
```

在 bert_predict.py 中指定下面三个参数就能加载训练好的模型完成预测:
```python
VOCAB_FILE = './pretrainmodel/vocab.txt'
LABEL_FILE = './output/label2id.pkl'
EXPORT_PATH = './export_models/1581318324'
```

代码参考: 
- [bert-chinese-ner](https://github.com/ProHiryu/bert-chinese-ner)
- [BERT-NER](https://github.com/kyzhouhzau/BERT-NER)

## 1.train file

训练数据的格式和CRF++的训练数据一致,每一列用`\t`分隔，每一个句子用`\n`分隔

exmpale:

```
北	B	
京	E	
欢	B	
迎	E	
你	S	
```


```bash
cd train
vim main.sh
```

在```main.sh```中指定以下参数

```bash

# train file
TRAIN_FILE=./datasets/demo/train.txt
# dev file
DEV_FILE=./datasets/demo/dev.txt
# test file
TEST_FILE=./datasets/demo/test.txt

# data out dir
DATA_OUT_DIR=./datasets/demo

# model save dir 
MODEL_OUT_DIR=./results/demo_seg/

# label tag column index
TAG_INDEX=1

# max length of sentlen
MAX_LENGTH=100

```

## 2.embeding

编译word2vec

```bash
cd third_paty && make

```

在```main.sh```中指定 word2vec 路径

```
WORD2VEC=./third_party/word2vec/word2vec
````

默认使用word2vec 训练字向量

```bash
./main.sh vec
```

## 3.map file
这一步产生需要的映射文件

```bash
./main.sh map
```

## 4.tfrecord
为了处理好内存，先把训练数据转换成tfrecord格式

```bash
./mainsh data
```

## 5.train 
```bash
./main.sh train
```

## export model
训练好的模型导出成.pb文件，导出路径见 ```main.sh``` 中```MODEL_PATH```
下面这个命令会导出最新的模型文件

```bash
./main.sh export
```

## load model
训练好模型，现在可以直接调用
```python

import fool

map_file = "./datasets/demo/maps.pkl"
checkpoint_ifle = "./results/demo_seg/modle.pb"

smodel = fool.load_model(map_file=map_file, model_file=checkpoint_ifle)
tags = smodel.predict(["北京欢迎你", "你在哪里"])
print(tags)

```

## 注

如果需要新增新的特征，要修改很多代码，请看懂后随意修改，**没有东西是完全正确的当然也包括我的代码**。