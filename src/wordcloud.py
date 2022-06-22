from PIL import Image
import numpy as np

# 用wordcloud做词云展示
from wordcloud import WordCloud, ImageColorGenerator

import jieba

# 读取标点符号库
f=open("stopwords.txt","r")
stopwords={}.fromkeys(f.read().split("\n"))
f.close()

print(stopwords)

#加载用户自定义词典
jieba.load_userdict("user_dict.txt")
#强调特殊名词
""" jieba.suggest_freq(('996'), True)
jieba.suggest_freq(('劳动法'), True)
jieba.suggest_freq(('老板'), True)
jieba.suggest_freq(('程序员'), True)
 """

file_name =r'C:\Users\Administrator\Desktop\jieba例子.txt' 
with open(file_name,'r') as f:
    content = f.read()

#这里我定义了一个函数cut_word()：
def cut_word():
    segment=[]
    # 保存分词结果
    segs=jieba.cut(content) 
    # 对整体进行分词
    for seg in segs:
        if len(seg) > 1 and seg != '\r\n':
    # 如果说分词得到的结果不是单字，且不是换行符，则加入到数组中
            segment.append(seg)
    return segment

print(cut_word())
 
segs=jieba.cut(text)

mytext_list=[]
#文本清洗
for seg in segs:
    if seg not in stopwords and seg!=" ":
        mytext_list.append(seg.replace(" ",""))
cloud_text=",".join(mytext_list) 
print(len(mytext_list))

#加载背景图片
cloud_mask = np.array(Image.open(r"background.jpg"))
#忽略显示的词

#生成wordcloud对象
wc = WordCloud(background_color="white", 
    mask=cloud_mask,  # 背景蒙版
    max_words=2000,  # 最大词数
    font_path="C:\Windows\Fonts\STFANGSO.TTF",
    min_font_size=15,  # 字体最小值
    max_font_size=60,   # 字体最大值
    # width=400, 
    stopwords=None)
wc.generate(cloud_text)
wc.to_file("pic.png")