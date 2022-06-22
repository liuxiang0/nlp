#jieba example6: 进行中文文章分词实现词云图与TOP词频统计

import docx
import jieba
from scipy.misc import imread
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud, ImageColorGenerator

versionN = 6  # 版本
filePath01 = r'F://data_temp/wordTest01.docx'  # 源文件路径
filePath02 = r'F://data_temp/wordCut-v{0}.txt'.format(versionN)   # 分词结果文件保存路径
filePath03 = r'F://data_temp/wordCount-v{0}.txt'.format(versionN)  # # 分词词频统计结果文件保存路径
filePath04 = r"F://data_temp/test.jpg"  # 词云图背景图片
filePath05 = r'F://data_temp/wordCloud-v{0}.jpg'.format(versionN)  # 词云图保存路径
filePath06 = r'F://data_temp/全新硬笔行书简体.ttf'  # 字体文件
filePath07 = r'F://data_temp/wordCountBar-v{0}.jpg'.format(versionN)  # TOP词频图保存路径
 
file01 = docx.Document(filePath01)
docText01 = ''
for i in file01.paragraphs:
    docText01 = docText01 + i.text
segList = '/'.join(jieba.cut(docText01, cut_all=False))  # cut_all=False 精确分词,分词符号为/
with open(filePath02, 'a', encoding='utf-8') as f1:  # 保存分词结果
    f1.write(segList)
    f1.close()
wordList = segList.split('/')
arr = np.array(wordList)
keyUse = np.unique(arr)
wordDict = {}
for i in keyUse:
    mask = (arr == i)  # return like this [ True False ... False False  True]
    arr_new = arr[mask]  # get the True index element
    v = arr_new.size  # 计数 count the size of i
    wordDict[i] = v  # 赋值 assignment index i of dict
wordDictSorted = sorted(wordDict.items(), key=lambda item: item[1], reverse=True)  # reverse=True 按value值降序排列
PunctuationS = ['，', '。', '?', '、', ' ', '“', '”', '：', '（', '）',
                '.', '', '', '', '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '', '', '', '', '']
with open(filePath03, 'a', encoding='utf-8') as f2:
    for i in wordDictSorted:
        if i[0] not in PunctuationS and len(i[0]) > 1:
            f2.write('{0}|{1}\n'.format(i[0], i[1]))
        else:
            continue
    f2.close()
 
color_mask = imread(filePath04)  # 读取背景图片,注意路径
wc = WordCloud(
    scale=6,   # 越大分辨率越高
    font_path="simkai.ttf",  # 设置字体，不指定就会出现乱码，注意字体路径
    #font_path=path.join(d,'simsun.ttc'),
    background_color='white',  # 设置背景色
    mask=color_mask,  # 词云形状
    max_words=2000,  # 允许最大词汇
    max_font_size=60  # 最大号字体
)
wc.generate(segList)  # 产生词云
image_colors = ImageColorGenerator(color_mask)  # 从背景图片生成颜色值
wc.to_file(filePath05)  # 保存图片
plt.figure()  # 修复不显示图片的bug
plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")  # 实现词云图片按照图片颜色取色
plt.axis("off")  # 关闭坐标轴
plt.show()
 
# 画出词频统计条形图，用渐变颜色显示，选取前N个词频
fig, ax = matplotlib.pyplot.subplots()  # fig：matplotlib.figure.Figure对象  ax：Axes(轴)对象或Axes(轴)对象数组
myFont = matplotlib.font_manager.FontProperties(fname=filePath06)  # 指定一个ttf字体文件作为图表使用的字体
# 默认状态下，matplotlb无法在图表中使用中文
 
words = []
counts = []
topN = 30
wordCount01 = open(filePath03, 'r', encoding='utf-8')
for i in wordCount01:
    words.append(i.split('|')[0])
    counts.append(int(i.split('|')[1].strip(r'\n')))
 
 
# 这里是为了实现条状的渐变效果，以该色号为基本色实现渐变效果
colors = ['#FA8072']
for i in range(len(words[:30]) - 1):
    colors.append('#FA{0}'.format(int(colors[-1][3:]) - 1))
 
rectS = ax.barh(np.arange(topN), counts[:topN], align='center', color=colors)  # 绘制横向条形图
# 修改Y轴的刻度
ax.set_yticks(np.arange(topN))  # 设置刻度值
ax.set_yticklabels(words[:topN], fontproperties=myFont)  # 因为已经排序好,所以直接取前三十个即可，用词替换刻度值
ax.invert_yaxis()  # 翻转Y坐标轴
ax.set_title('文章中的高频词汇', fontproperties=myFont, fontsize=17)  # 设置标题
ax.set_xlabel(u"出现次数", fontproperties=myFont)  # 设置X轴标题
for rect in rectS:
    width = rect.get_width()
    ax.text(1.03 * width, rect.get_y() + rect.get_height()/2., '%d' % int(width), ha='center', va='center')
plt.rcParams['figure.figsize'] = (8.0, 4.0)  # 设置figure_size尺寸
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.savefig(filePath07)
# 不知道为什么会报错ValueError: setting an array element with a sequence.,但是保存图片成功
plt.show()
