import sys
import os
import re
import chardet
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from kernel_word2vec import *
# from gensim.models import word2vec

word_tokenizer = RegexpTokenizer(r'\w+')
word_prog = re.compile('[A-Za-z]+')
cached_stop_words = stopwords.words("english")

class MyCorpus(object):
    """a memory-friendly iterator"""
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            fpath = os.path.join(self.dirname, fname)
            if not os.path.isfile(fpath) or fname[0] == '.':
                continue
            try:
                with open(fpath) as fp:
                    text = fp.read().lower()
                    try:
                        sentences = sent_tokenize(text)
                    except:
                        sentences = sent_tokenize(text.decode(chardet.detect(text)['encoding']))
                    for sent in sentences:
                        words = word_tokenizer.tokenize(sent)
                        words = [word for word in words if word not in cached_stop_words]
                        yield words
            except Exception as e:
                print e
                sys.exit()
            else:
                fp.close()

    def __len__(self):
        return sum(1 for fname in os.listdir(self.dirname) if os.path.isfile(os.path.join(self.dirname, fname)) and not fname[0] == '.')

def save_word_emb(vocab_path, emb_path, model):
    try:
        f1 = open(vocab_path, 'w')
        f2 = open(emb_path, 'w')
        for word in model.vocab.keys():
            f1.write('%s\n' % word)
            f2.write('%s\n' % ' '.join([str(x) for x in model[word]]))
    except Exception as e:
        print e
        return
    else:
        f1.close()
        f2.close()

def visual_tsne(model, out_file, watch_list):
    emb = []
    for x in model.vocab.keys():
        if x in watch_list:
            emb.append([y for y in model[x]])
    # emb = [[y for y in model[x]] for x in model.vocab.keys()]
    tsne = TSNE(n_components=2, random_state=0)
    reduced_emb = tsne.fit_transform(emb)
    np.savetxt(out_file, reduced_emb)
    return reduced_emb

def plot_tsne_emb(model, out_file, word_freq, min_count=10, ntop=1000):
    vocab = []
    emb = []
    count = 0
    if ntop == None:
        ntop = len(word_freq)
    for word, freq in word_freq.iteritems():
        if freq < min_count or word.isdigit():
            continue
        vocab.append(word)
        emb.append([y for y in model[word]])
        count += 1
        if count >= ntop:
            break
    print "totally %s words" % len(vocab)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    reduced_emb = tsne.fit_transform(emb)

    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1])
    for label, x, y in zip(vocab, reduced_emb[:, 0], reduced_emb[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.title("min_count=%s"%min_count)
    plt.show()

def PCA():
    pass

def KernelPCA(gamma):
    pass

if __name__ == '__main__':
    usage = 'python run.py [corpus_path] [out_emb_model_path] [vocab_path] [word_emb_path]'
    try:
        corpus_path = sys.argv[1]
        out_emb_model_path = sys.argv[2]
        # vocab_path = sys.argv[3]
        # word_emb_path = sys.argv[4]
        # tsne_emb_path = sys.argv[5]
        # word_freq_path = sys.argv[6]
    except:
        print usage
        sys.exit()
    if os.path.exists(out_emb_model_path):
        model = Word2Vec.load_word2vec_format(out_emb_model_path, binary=True)
    else:
        sentences = MyCorpus(corpus_path)
        model = Word2Vec(sentences, size=200, gamma=.005, min_count=100, workers=multiprocessing.cpu_count(), sg=1, negative=5, iter=1)
        print "total training time: %s" % model.total_train_time
        model.save_word2vec_format(out_emb_model_path, binary=True)

    # save_word_emb(vocab_path, word_emb_path, model)
    # import pdb;pdb.set_trace()
    # import json
    # visual_tsne(model, tsne_emb_path, vocab.keys())
    # with open(word_freq_path, 'r') as f:
    #     word_freq = json.load(f)
    #     plot_tsne_emb(model, tsne_emb_path, word_freq, min_count=100, ntop=100)

