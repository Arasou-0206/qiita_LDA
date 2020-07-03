import gensim
import re
import json
import csv
import pyLDAvis
import pyLDAvis.gensim

data = list(csv.reader(open('result.txt')))


dictionary = gensim.corpora.Dictionary(data)
dictionary.filter_extremes(no_below=3, no_above=0.8)
corpus = [dictionary.doc2bow(t) for t in data]
print('vocab size: ', len(dictionary))

#LDAvis
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=8, random_state=0)
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
vis
pyLDAvis.save_html(vis, 'LDAvis_output.html')