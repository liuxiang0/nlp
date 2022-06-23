
## 停用词

准备一个停用词的文本，停用词就是遇到这个词就跳过，如“了”、“的”、“吧嗒”等一些*没有意义的词汇和符号*。
我使用的停用词为哈工大停用词库，采用Jieba分词并去停用词。

停用词表 stopwords.txt，四川大学和哈工大的自己选择。

百度网盘地址在链接: https://pan.baidu.com/s/1KBkOzYk-wRYaWno6HSOE9g 提取码: 4sm6

注意：如果是gbk编码，请用编辑软件重新保存为 utf-8 编码。vscode中 只要先用gbk编码正常打开，再用utf-8编码保存即可。

~~~python
import jieba
 
# 创建停用词列表
def stopwordslist():
    stopwords = [line.strip() for line in open('./data/stopwords.txt',encoding='UTF-8').readlines()]
    return stopwords
 
# 对句子进行中文分词
def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    # print("正在分词")
    sentence_depart = jieba.cut(sentence.strip())
    # 创建一个停用词列表
    stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr
 
# 给出文档路径
filename = "./data/all_data.txt"
outfilename = "./data/out.txt"
inputs = open(filename, 'r', encoding='UTF-8')
outputs = open(outfilename, 'w', encoding='UTF-8')
 
# 将输出结果写入ou.txt中
for line in inputs:
    a = line.split()
    line = a[1]
    label = a[0]
    line_seg = seg_depart(line)
    outputs.write(label + '\t'+ line_seg + '\n')
    # print("-------------------正在分词和去停用词-----------")
outputs.close()
inputs.close()
print("删除停用词和分词成功！！！")
~~~
