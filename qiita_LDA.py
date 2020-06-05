import json
import itertools
import gensim
import random
import urllib.request 
from pathlib import Path
import re
import numpy as np
from collections import Counter
from sklearn import datasets
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.special import digamma

from pyquery import PyQuery as pq
p = re.compile(r"<[^>]*?>")
with open('test2.json', 'r') as f :
    jsn = json.load(f)
#print(len(jsn))

#データ処理#
docs = [" "]
stopwords = []

for v in range(500):
    docs.extend(p.sub("",jsn[v]['rendered_body']).split())
    #print(docs)

texts = [
    [w for w in doc.lower().split()]
    for doc in docs
]


#単語出現頻度#
count = Counter(w for doc in texts for w in doc)
print(count.most_common()[:10])
y = [i[1] for i in count.most_common()]
#plt.plot(y)
#plt.savefig('out_graph.png', dpi=300, orientation='portrait', transparent=False, pad_inches=0.0)
#plt.savefig('out_graph2.pdf', orientation='portrait', transparent=False, bbox_inches=None, frameon=None)
#plt.show()

#辞書作成#
dictionary = gensim.corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
#print(corpus[1])

#トピックモデル学習
num_topics = 20
 
lda = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    num_topics=num_topics,
    id2word=dictionary
)

#表示
plt.figure(figsize=(30,30))
for t in range(lda.num_topics):
    plt.subplot(5,4,t+1)
    x = dict(lda.show_topic(t,200))
    im = WordCloud().generate_from_frequencies(x)
    plt.imshow(im)
    plt.axis("off")
    plt.title("Topic #" + str(t))

plt.savefig('out_graph2.png', dpi=300, orientation='portrait', transparent=False, pad_inches=0.0)
plt.savefig('out_graph2.pdf', orientation='portrait', transparent=False, bbox_inches=None, frameon=None)
plt.show()