#jieba example1: 分词，word segment

import jieba

"""
jieba.cut_for_search方法接受一个参数：需要分词的字符串,
该方法适合用于搜索引擎构建倒排索引的分词，粒度比较细 
注意：待分词的字符串可以是gbk字符串、utf-8字符串或者unicode 
jieba.cut以及jieba.cut_for_search返回的结构都是一个可迭代的generator，
可以用list(jieba.cut(...))转化为list ，
也可以使用for循环来获得分词后得到的每一个词语(unicode)。
"""

# 1. 全模式cut_all=True
seg_list = jieba.cut("伟大的北京天安门", cut_all=True) 
print("Full Mode:", "/ ".join(seg_list) )
# 2. 精确模式，缺省模式default
seg_list = jieba.cut("伟大的北京天安门", cut_all=False) 
print("Default Mode:{}".format(list(seg_list))) 
# 3. 默认是精确模式
seg_list = jieba.cut("这里是伟大的北京天安门")    
print(", ".join(seg_list)  )

# 4. 搜索引擎模式
seg_list = jieba.cut_for_search("这里是伟大的北京天安门，伟大的中华人民共和国！") 
print("搜索引擎:{}".format(", ".join(seg_list) ))

# 5. 缺省模式，结果为列表
seg_list = jieba.lcut("这里是伟大的北京天安门，伟大的中华人民共和国！") 
print("缺省模式下的列表lcut:{}".format(seg_list))

# 6. 搜索模式，结果为列表
seg_list = jieba.lcut_for_search("这里是伟大的北京天安门，伟大的中华人民共和国！") 
print("搜索模式下的列表lcut:{}".format(seg_list))
