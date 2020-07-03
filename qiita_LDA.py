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
import matplotlib
import matplotlib.pylab as plt
from wordcloud import WordCloud
from scipy.special import digamma
import MeCab
from pyquery import PyQuery as pq
from tqdm import tqdm
import pyLDAvis
import pyLDAvis.gensim


p = re.compile(r"<[^>]*?>")
with open('test2.json', 'r') as f :
    jsn = json.load(f)
#print(len(jsn))

#データ処理#
docs = [" "]
stopwords = []

for v in range(5):
    docs.extend(p.sub("",jsn[v]['rendered_body']).split())
    #print(docs)

texts = [
    [w for w in doc.lower().split()]
    for doc in docs
]


#単語出現頻度#
count = Counter(w for doc in texts for w in doc)
#print(count.most_common()[:10])
#y = [i[1] for i in count.most_common()]
#plt.plot(y)
#plt.savefig('out_graph.png', dpi=300, orientation='portrait', transparent=False, pad_inches=0.0)
#plt.savefig('out_graph2.pdf', orientation='portrait', transparent=False, bbox_inches=None, frameon=None)
#plt.show()

#辞書作成#
dictionary = gensim.corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
#頻出を絞りたい
#num_tokens = len(count.most_common())
#N = int(num_tokens*0.05)
#max_frequency = count.most_common()[N][1]
#corpus = [[w for w in doc if max_frequency > w[1] >= 3] for doc in corpus]

#lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=8, random_state=0)
#pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
#vis
#pyLDAvis.save_html(vis, 'pyldavis_output.html')

start = 2
limit = 30
step = 2

coherence_vals = []
perplexity_vals = []

for n_topic in range(start, limit, step):

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topic, random_state=0)
    perplexity_vals.append(np.exp2(-lda_model.log_perplexity(corpus)))
    coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_vals.append(coherence_model_lda.get_coherence())

fig, ax1 = plt.subplots(figsize=(12,5))
x = range(start, limit, step)

c1 = 'gray'
ax1.plot(x, coherence_vals, 'o-', color=c1)
ax1.set_xlabel('Num Topics')
ax1.set_ylabel('Coherence', color=c1); ax1.tick_params('y', colors=c1)

c2 = 'green'
ax2 = ax1.twinx()
ax2.plot(x, perplexity_vals, 'o-', color=c2)
ax2.set_ylabel('Perplexity', color=c2); ax2.tick_params('y', colors=c2)

ax1.set_xticks(x)
fig.tight_layout()
plt.grid()
plt.show()
