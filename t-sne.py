#!/usr/bin/env python3
"""
Author      : Yukun Feng
Date        : 2018/07/23
Email       : yukunfg@gmail.com
Description : T-SNE to visualize word vectors
(codes are from some posts in the Internet)
https://medium.com/@aneesha/using-tsne-to-plot-a-subset-of-similar-words-from-word2vec-bb8eeaea6229
"""

import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main():
    embeddings_file = sys.argv[1]
    wv, vocabulary = load_embeddings(embeddings_file)
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv[:1000, :])

    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(
            label, xy=(x, y),
            xytext=(0, 0), textcoords='offset points'
        )
    plt.show()


def load_embeddings(file_name):
    "load embeddings from file"
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in f_in])
    wv = np.loadtxt(wv)
    return wv, vocabulary


if __name__ == '__main__':
    main()
