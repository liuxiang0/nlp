# jieba example4: 关键词提取
"""
核心调用方法：jieba.analyse.extract_tags(sentence,topK)  
#需要先import jieba.analyse 
sentence为待提取的文本 ， topK为返回几个TF/IDF权重最大的关键词，默认值为20 
"""

import sys  
sys.path.append('../')  
import jieba  
import jieba.analyse  
from optparse import OptionParser  
USAGE = "usage: python extract_tags.py [file name] -k [top k]"  
parser = OptionParser(USAGE)  
parser.add_option("-k", dest="topK")  
opt, args = parser.parse_args()  
#'''
if len(args) < 1:  
    print(USAGE)  
    sys.exit(1)  
#'''   
file_name = args[0]
#file_name=u"story.txt"  
if opt.topK is None:  
    topK = 10  
else:  
    topK = int(opt.topK)   
content = open(file_name, 'r').read()  
tags = jieba.analyse.extract_tags(content, topK=topK)  
print(",".join(tags) )