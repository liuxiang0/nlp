# jieba07-stopword.py 去停用词案例

from collections import Counter
import jieba
 
 
# jieba.load_userdict('userdict.txt')
# 创建停用词list, open的 encoding='utf-8' 为缺省
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]
    return stopwords
 
# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('./stopword/stopwords.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr
 
 
inputs = open('splitter.txt', 'r', encoding='utf-8')  # 加载要处理的文件的路径
outputs = open('./output/result.txt', 'w', encoding='utf-8')  # 加载处理后的文件路径
for line in inputs:
    line_seg = seg_sentence(line)  # 这里的返回值是字符串
    outputs.write(line_seg)
outputs.close()
inputs.close()
# WordCount
with open('./output/result.txt', 'r',encoding='utf-8') as fr:  # 读入已经去除停用词的文件
    data = jieba.cut(fr.read())
data = dict(Counter(data))
 
with open('./output/wordcount.txt', 'w') as fw:  # 读入存储wordcount的文件路径
    for k, v in data.items():
        if len(k)>1: #此处根据需求进行修改
            if not k.isdecimal(): # 自动过滤一个汉字, 数字和空白字符。
                fw.write('%s,%d\n' % (k, v))
        