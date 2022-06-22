#! -*- coding:utf-8 -*-

import jieba
import jieba.analyse

# 待分词的文本路径
sourceTxt = 'wenben1.txt '

# 分好词后的文本路径
targetTxt = 'wenben2.txt'

with open(sourceTxt, 'r') as sourceFile, open(targetTxt, 'a+') as targetFile:
    for line in sourceFile:
        seg = jieba.cut(line.strip(), cut_all = False)
        # 分好词之后之间用空格隔断
        output = ' '.join(seg)
        targetFile.write(output)
        targetFile .write('\n')

print('写入成功！')

with open(targetTxt, 'r') as file:
    text = file.readlines()

    """
    几个参数解释：
    * text : 待提取的字符串类型文本
    * topK : 返回TF-IDF权重最大的关键词的个数，默认为20个
    * withWeight : 是否返回关键词的权重值，默认为False
    * allowPOS : 包含指定词性的词，默认为空
    """

    keywords = jieba.analyse.extract_tags(str(text), topK = 10, withWeight=True, allowPOS=())
    print(keywords)

print('提取完毕！')
