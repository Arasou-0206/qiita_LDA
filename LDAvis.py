import gensim
import re
import json
import csv
import pyLDAvis
import pyLDAvis.gensim

with open('Japanese.txt', 'r', encoding='UTF-8') as fd:
    stop_words = fd.read()

text = list(csv.reader(open('result.txt', encoding='UTF-8')))

for i in range(len(text)):
    for j in range(len(text[i])):
        text[i][j] = re.sub(r'[!-@]', "", text[i][j])
        text[i][j] = re.sub(r'[{-~]', "", text[i][j])
        text[i][j] = text[i][j].lower()

data = []
for i in range(len(text)):
    data.append([word for word in text[i]
                 if word not in stop_words and len(word) >= 2])

with open('fil_data.pkl', 'wb') as w:
    pickle.dump(data[::2], w)

dictionary = gensim.corpora.Dictionary(data)
dictionary.filter_extremes(no_below=3, no_above=0.8)
corpus = [dictionary.doc2bow(t) for t in data]
print('vocab size: ', len(dictionary))

#LDAvis
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=6, random_state=0)
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
vis
pyLDAvis.save_html(vis, 'LDAvis_output.html')
