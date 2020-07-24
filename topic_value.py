import gensim
import re
from gensim.models import LdaModel
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import pickle
from collections import defaultdict

with open('dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)
with open('fil_data.pkl', 'rb') as f:
    data = pickle.load(f)
with open('corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)
print('set data OK')
print(len(data))

lda = LdaModel(corpus=corpus, num_topics=6, id2word=dictionary)
score_by_topic = []
for unseen_doc, raw_train_text in zip(corpus, data):
    score_by_topic.append(lda[unseen_doc])

# with open('max_topic.pkl', 'wb') as w:
#    pickle.dump(max_topic, w)
