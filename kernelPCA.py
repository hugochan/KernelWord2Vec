import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from kernel_word2vec import *

np.random.seed(0)

def get_dictionary(dictionary_path):
    vocabs = []
    try:
        with open(dictionary_path, 'r') as f:
            for line in f:
                vocabs.extend(line.strip('\n ').replace('-', ' ').replace('\xe2\x80\x93', ' ').replace('\xe2\x80\x99', ' ').split())
    except Exception as e:
        print e
        return
    else:
        f.close()
        return list(set(vocabs))

def plot_kpca_emb(model, word_freq, dictionary, min_count=10, ntop=1000):
    vocab = []
    emb = []
    count = 0
    if ntop == None:
        ntop = len(word_freq)
    for word, freq in sorted(word_freq.iteritems(), key=lambda d:d[1], reverse=True):
        if freq < min_count or word.isdigit() or not str(word) in dictionary:
            continue
        vocab.append(word)
        emb.append([y for y in model[word]])
        count += 1
        if count >= ntop:
            break
    print "totally %s words" % len(vocab)

    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=False, gamma=.01)
    X_kpca = kpca.fit_transform(emb)
    # X_back = kpca.inverse_transform(X_kpca)
    # plt.plot(X_kpca[:, 0], X_kpca[:, 1], "ro")
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1])
    for label, x, y in zip(vocab, X_kpca[:, 0], X_kpca[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.title("Kernel word embeddings projected by KPCA (top %s words)"%len(vocab))
    plt.xlabel("1st principal component in space induced by $\phi$")
    plt.ylabel("2nd component")

    plt.show()


if __name__ == '__main__':
    usage = 'python kernelPCA.py [model_path] [word_freq_path] [dictionary_path]'
    try:
        model_path = sys.argv[1]
        word_freq_path = sys.argv[2]
        dictionary_path = sys.argv[3]
    except:
        print usage
        sys.exit()


    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    with open(word_freq_path, 'r') as f:
        word_freq = json.load(f)

    dictionary = get_dictionary(dictionary_path)

    plot_kpca_emb(model, word_freq, dictionary, min_count=500, ntop=100)



    # pca = PCA()
    # X_pca = pca.fit_transform(X)
