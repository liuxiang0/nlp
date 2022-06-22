# jieba example5: 并行分词
"""
原理：将目标文本按行分隔后，把各行文本分配到多个python进程并行分词，
然后归并结果，从而获得分词速度的可观提升 
基于python自带的multiprocessing模块，目前暂不支持windows 
    jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数  
    jieba.disable_parallel() # 关闭并行分词模式 
实验结果：在4核3.4GHz Linux机器上，对金庸全集进行精确分词，
获得了1MB/s的速度，是单进程版的3.3倍。
"""

import urllib3
import sys,time  
import sys  
sys.path.append("../../")  
import jieba  
jieba.enable_parallel(4)  
  
url = sys.argv[1]  
content = open(url,"rb").read()  
t1 = time.time()  
words = list(jieba.cut(content))  
  
t2 = time.time()  
tm_cost = t2-t1  
  
with open("05parallel.log","wb") as log_f:
    for w in words:  
        log_f.write(w.encode("utf-8")+"/")
  
print('speed' , len(content)/tm_cost, " bytes/second" )