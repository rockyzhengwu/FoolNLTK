# FoolNLTK
A Chinese word processing toolkit

[Chinese document](./README_CH.md)
## Features
* Although not the fastest, FoolNLTK is probably the most accurate open source Chinese word segmenter in the market
* Trained based on the [BiLSTM model](http://www.aclweb.org/anthology/N16-1030 )
* High-accuracy in participle, part-of-speech tagging, entity recognition
* User-defined dictionary
* Ability to self train models
* Allows for batch processing


## Getting Started

*** 2020/2/16 ***  update: use bert model train and export model to deploy, [chinese train documentation](./train/README.md)



To download and build FoolNLTK, type:

```bash
get clone https://github.com/rockyzhengwu/FoolNLTK.git
cd FoolNLTK/train

```
For detailed [instructions](./train/README.md)

* Only tested in Linux Python 3 environment. 


### Installation
```bash
pip install foolnltk
```


## Usage Intructions

##### For Participles:



```
import fool

text = "一个傻子在北京"
print(fool.cut(text))
# ['一个', '傻子', '在', '北京']
```

For participle segmentations, specify a ```-b``` parameter to increase the number of lines segmented every run.  

```bash
python -m fool [filename]
```

###### User-defined dictionary
The format of the dictionary is as follows: the higher the weight of a word, and the longer the word length is, 
the more likely the word is to appear. Word weight value should be greater than 1。 

```
难受香菇 10
什么鬼 10
分词工具 10
北京 10
北京天安门 10
```
To load the dictionary:

```python
import fool
fool.load_userdict(path)
text = ["我在北京天安门看你难受香菇", "我在北京晒太阳你在非洲看雪"]
print(fool.cut(text))
#[['我', '在', '北京', '天安门', '看', '你', '难受', '香菇'],
# ['我', '在', '北京', '晒太阳', '你', '在', '非洲', '看', '雪']]
```

To delete the dictionary
```python
fool.delete_userdict();
```



##### POS tagging

```
import fool

text = ["一个傻子在北京"]
print(fool.pos_cut(text))
#[[('一个', 'm'), ('傻子', 'n'), ('在', 'p'), ('北京', 'ns')]]
```


##### Entity Recognition
```
import fool 

text = ["一个傻子在北京","你好啊"]
words, ners = fool.analysis(text)
print(ners)
#[[(5, 8, 'location', '北京')]]
```

### Versions in Other languages
* [Java](https://github.com/rockyzhengwu/JFoolNLTK)

#### Note
* For any missing model files, try looking in ```sys.prefix```, under ```/usr/local/```
