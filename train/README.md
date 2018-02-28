# FoolNLTK-train
FoolNLTK training process

在新的分支idcnn里添加了[dilated-cnn](https://arxiv.org/abs/1702.02098)的训练代码
感谢[原作者的代码](https://github.com/iesl/dilated-cnn-ner)

主要有以下几个文件:
```cnn.py``` 模型文件
```cnn_train.py```训练代码
```cnn_main.sh```　整合了一些命令到shell中

另外我对比了LSTM和CNN的速度CNN有提高，但并没有到惊人的程度

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