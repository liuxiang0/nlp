"""jieba08-wordcloud.py: 
https://alvinntnu.github.io/python-notes/corpus/jieba.html
"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]
    return stopwords

stopwords =["包括","但","另"] #stopwordslist('./stopword/stopwords.txt') 

words2 = "據《日經亞洲評論》網站報導，儘管美國總統川普發起了讓美國製造業回歸的貿易戰，但包括電動汽車製造商特斯拉在內的一些公司反而加大馬力在大陸進行生產。另據高盛近日發布的一份報告指出，半導體設備和材料以及醫療保健領域的大多數公司實際上正擴大在大陸的生產，許多美國製造業拒絕「退出中國」。"

wf = dict(sorted(Counter(words2).items(), key=lambda x:x[1], reverse=True))

wc = WordCloud(background_color='white',
               #font_path='/System/Library/Fonts/STHeiti Medium.ttc',
               random_state=10,
               max_font_size=None,
               stopwords=stopwords) ## stopwords not work when wc.genreate_from_frequencies

wc.generate_from_frequencies(frequencies=wf)

plt.figure(figsize=(15,15))
plt.imshow(wc)
plt.axis("off")
plt.show()