# FoolNLTK
中文处理工具包

## 特点
* 可能不是最快的开源中文分词，但很可能是最准的开源中文分词
* 基于[BiLSTM模型](http://www.aclweb.org/anthology/N16-1030 )训练而成
* 包含分词，词性标注，实体识别,　都有比较高的准确率
* 用户自定义词典



## Install
```bash
pip install foolnltk
```

### 使用说明

##### 分词



```
import fool

text = "一个傻子在北京"
print(fool.cut(text))
# ['一个', '傻子', '在', '北京']
```
命令行分词
```bash
python -m fool [filename]
```

###### 用户自定义词典
词典格式格式如下，词的权重越高，词的长度越长就越越可能出现,　权重值请大于1
```
难受香菇 10
什么鬼 10
分词工具 10
北京 10
北京天安门 10
```
加载词典

```python
import fool
fool.load_userdict(path)
text = "我在北京天安门看你难受香菇"
print(fool.cut(text))
# ['我', '在', '北京天安门', '看', '你', '难受香菇']
```
删除词典
```python
fool.delete_userdict();
```



##### 词性标注

```
import fool

text = "一个傻子在北京"
print(fool.pos_cut(text))
#[('一个', 'm'), ('傻子', 'n'), ('在', 'p'), ('北京', 'ns')]
```


##### 实体识别
```
import fool 

text = "一个傻子在北京"
words, ners = fool.analysis(text)
print(ners)
#[(5, 8, 'location', '北京')]
```
#### 注意
* 暂时只在Python3 Linux 平台测试通过

