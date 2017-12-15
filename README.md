# FoolNLTK
中文处理工具包

## 特点
* 基于[BiLSTM模型](http://www.aclweb.org/anthology/N16-1030 )训练而成
* 包含分词，词性标注，实体识别,　都有比较高的准确率
* 暂时不支持外部词典



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

