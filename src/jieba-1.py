#! -*- coding:utf-8 -*-

import jieba
import jieba.analyse
import time

# 待分词的文本路径
sourceTxt = 'splitter.txt'

# 分好词后的文本路径
targetTxt = 'words.txt'
t1=time.time()

with open(sourceTxt, 'r', encoding='utf-8') as sourceFile, open(targetTxt, 'a+',encoding='utf-8') as targetFile:
    for line in sourceFile:
        seg = jieba.cut(line.strip(), cut_all = False) #用精确模式进行中文分词
        # 分好词之后之间用空格隔断，也可以用‘/’，此处为迭代器generator
        output = '/'.join(seg)
        targetFile.write(output) # 将分词结果写入文件。
        targetFile .write('\n')

print('jieba中文分词写入{}成功！'.format(targetTxt))

with open(targetTxt, 'r') as file:
    text = file.readlines()

    """
    几个参数解释：
    * text : 待提取的字符串类型文本
    * topK : 返回TF-IDF权重最大的关键词的个数，默认为20个
    * withWeight : 是否返回关键词的权重值，默认为False
    * allowPOS : 包含指定词性的词，默认为空
    """

    keywords = jieba.analyse.extract_tags(str(text), topK = 30, withWeight=True, allowPOS=())
    print(keywords)

print('提取完毕！')
t2=time.time()
print("中文分词及词性标注完成，耗时："+str(t2-t1)+"秒。") # 反馈结果

"""
#coding=utf-8
import jieba
import jieba.posseg as pseg
import time

f=open("t_with_splitter.txt","r") #读取文本
string=f.read().decode("utf-8")
words = pseg.cut(string) #进行分词
result="" #记录最终结果的变量
for w in words:
   result+= str(w.word)+"/"+str(w.flag) #加词性标注
f=open("t_with_POS_tag.txt","w") #将结果保存到另一个文档中
f.write(result)
f.close()


"""