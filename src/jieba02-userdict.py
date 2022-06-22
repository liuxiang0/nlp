#jieba example2: 自定义词典

import sys  
sys.path.append("./")  
import jieba   
import jieba.posseg as pseg  
  
test_sent = "李小福是创新办主任也是云计算方面的专家;"  
test_sent += "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类型，python 的正则表达式是好用的"  
print("\n====下面结果未自定义词典====")
words = jieba.cut(test_sent) 
print("缺省模式分词:", "/ ".join(words)  )

print("\n====下面是自定义userdict分词====")
jieba.load_userdict("userdict.txt")
words = jieba.cut(test_sent) 
print("使用自定义词典后分词:", "/ ".join(words)  )    