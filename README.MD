# 基于词典正负情绪二分类任务
## 方法：词典法
### 实现方式：
#### 代码：text_emotion_classification.ipynb
基于已有词典（例如：Hownet词典或者大连理工中文情感词典）中的词语情感极性值计算得出的。
在情绪正负判别中被广泛使用到的是：Hownet词典是一个基于概念网络的大型中文语义知识库，其中包含了大量的词语情感极性信息。

实现方式：使用现成的第三方库(https://github.com/thunderhit/cnsenti)

在cnsenti中，每个词语会被赋予一个情感极性值，取值范围为-1到1，表示词语的负面情感程度和正面情感程度。根据每个词语的情感极性值，以及情感词前是否有强度副词的修饰作用以及是否有否定词的情感语义反转作用，cnsenti会计算出整个文本的情感极性评分。情感极性评分的取值范围为-1到1，越接近1表示越积极，越接近-1表示越消极，趋近于0表示中性。
# 基于bert模型进行情绪多分类任务
## 方法：直接使用已有模型进行情感预测
### 实现方式：
#### 代码：text_emotion_classification.ipynb
优点：方便，效果精确

缺点：难以匹配目标分类，模型分类可能并非任务分类
## 方法：使用prompt计算文本情感分数
### 实现方式：
#### 代码：text_emotion_classification.ipynb或predict.py中predict_by_prompt函数

## 方法：基于bert的微调
### 实现方式：
#### 代码：bert_fine_tune.py,predict_by_bert.py
在预训练模型中添加一些全连接层组成，最后接一个softmax层来输出分类结果
#### 可修改参数：max_length、batch_size、epochs以及learning_rate
# 基于RoBERTa模型进行情绪多分类任务
## 方法：基于XLMRoberta的微调
### 实现方式：
#### 代码：Roberta_fine_tune.py,predict_by_Robert.py
在预训练Roberta模型中添加一些全连接层组成，最后接一个softmax层来输出分类结果
#### 

# utility文件夹
## 作用：
用来存放功能函数，包括基于bert做微调的数据集加载函数和预测函数以及基于XLMRoberta微调的数据集加载函数和评估函数
# 基于LLM进行情绪多分类任务