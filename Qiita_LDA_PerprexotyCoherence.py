import gensim
import re
import csv
import pyLDAvis
import pyLDAvis.gensim
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle


def main():
    with open('dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    with open('fil_data.pkl', 'rb') as f:
        data = pickle.load(f)
    with open('corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)
    print('set data OK')

    # Perplexoty/Coherence
    start = 2
    limit = 10
    step = 2

    coherence_vals = []
    perplexity_vals = []

    for n_topic in range(start, limit, step):
        lda_model = gensim.models.ldamodel.LdaModel(
            corpus=corpus, id2word=dictionary, num_topics=n_topic, random_state=0)
        perplexity_vals.append(np.exp2(-lda_model.log_perplexity(corpus)))
        coherence_model_lda = gensim.models.CoherenceModel(
            model=lda_model, texts=data, dictionary=dictionary, coherence='c_v')
        coherence_vals.append(coherence_model_lda.get_coherence())

    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = range(start, limit, step)

    c1 = 'gray'
    ax1.plot(x, coherence_vals, 'o-', color=c1)
    ax1.set_xlabel('Num Topics')
    ax1.set_ylabel('Coherence', color=c1)
    ax1.tick_params('y', colors=c1)

    c2 = 'green'
    ax2 = ax1.twinx()
    ax2.plot(x, perplexity_vals, 'o-', color=c2)
    ax2.set_ylabel('Perplexity', color=c2)
    ax2.tick_params('y', colors=c2)

    ax1.set_xticks(x)
    fig.tight_layout()
    fig.savefig("PerprexotyCoherence(2-20).png")


if __name__ == '__main__':
    main()
