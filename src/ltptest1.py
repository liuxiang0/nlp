"""哈工大自然语言处理模块 ltp 学习

"""

from ltp import LTP

sentences = ["他叫汤姆去拿外衣。", "汤姆生病了。他去了医院。"]

# 1. 加载模型
ltp =LTP()
# ltp = LTP(path = "base|small|tiny")
# ltp = LTP(path = "tiny.tgz|tiny-tgz-extracted") # 其中 tiny-tgz-extracted 是 tiny.tgz 解压出来的文件夹

# 2. 分句
sents = ltp.sent_split(sentences)
print(sents)

# 3. 自定义词典
# user_dict.txt 是词典文件， max_window是最大前向分词窗口
ltp.init_dict(path="user_dict.txt", max_window=4)
# 也可以在代码中添加自定义的词语
ltp.add_words(words=["负重前行", "长江大桥", "不忘初心","砥砺前行"], max_window=4)

# 4. 分词Word Sense Disambugation (WSD)
segone = ["他叫汤姆去拿外衣。"]
segment, hidden = ltp.seg(segone)
# [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]

# 对于已经分词的数据
#segment, hidden = ltp.seg(["他/叫/汤姆/去/拿/外衣/。".split('/')], is_preseged=True)

# 5. 词性标注 Part-of-Speech Tagging(POSTag)
pos = ltp.pos(hidden)
print("pos:",pos)
# [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]
# [['r', 'v', 'nh', 'v', 'v', 'n', 'wp']]

# 6. 命名实体识别 Named Entity Recognition (NER)
ner = ltp.ner(hidden)
print("ner:",ner)
# [['他', '叫', '汤姆', '去', '拿', '外衣', '。']]
# [[('Nh', 2, 2)]]

tag, start, end = ner[0][0]
print(tag,":", "".join(segment[0][start:end + 1]))
# Nh : 汤姆

# 7. 语义角色标注 Semantic Role Labeling(SRL)
srl = ltp.srl(hidden)
print("srl1:",srl)
srl = ltp.srl(hidden, keep_empty=False)
print("srl2:",srl)

# 8. 依存句法分析
dep = ltp.dep(hidden)
print("dep:", dep)

# 9. 语义依存分析(树)
sdp = ltp.sdp(hidden, mode='tree')
print("sdp(Tree):", sdp)

# 10. 语义依存分析(图)
sdp = ltp.sdp(hidden, mode='graph')
print("sdp(Graph):",sdp)