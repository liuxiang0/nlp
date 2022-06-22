# jieba example3: 词性标注

import jieba.posseg as pseg

test_sent = "李小福是创新办主任也是云计算方面的专家;"  
test_sent += "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类型，python 的正则表达式是好用的"  
result = pseg.cut(test_sent)  
for w in result:  
    print(w.word, "/", w.flag, ", ",)
print("\n========")    