import gensim
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import pickle

with open('Japanese.txt','r',encoding='UTF-8') as fd:
    stop_words = fd.read()

text = list(csv.reader(open('result.txt',encoding='UTF-8')))
data = []
for i in range(len(text)):
    if len(text[i]) >= 1:
        data.append([word for word in text[i] if word not in stop_words])

dictionary = gensim.corpora.Dictionary(data)
dictionary.filter_extremes(no_below=3, no_above=0.8)
corpus = [dictionary.doc2bow(t) for t in data]


#Perplexoty/Coherence
def main() :
    start = 2
    limit = 20
    step = 2

    coherence_vals = []
    perplexity_vals = []

    for n_topic in tqdm(range(start, limit, step)):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topic, random_state=0)
        perplexity_vals.append(np.exp2(-lda_model.log_perplexity(corpus)))
        coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=data, dictionary=dictionary, coherence='c_v')
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
fig.savefig("PerprexotyCoherence.png")
