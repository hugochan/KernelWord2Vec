import sys
import os
import re
import chardet
import multiprocessing
import logging
import numpy as np
from gensim import utils
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from kernel_word2vec import *

word_tokenizer = RegexpTokenizer(r'\w+')
cached_stop_words = stopwords.words("english")

# parameters controlling what is to be computed: how many dimensions, window size etc.
DIM = 200
GAMMA = .001
MIN_COUNT = 5
DOC_LIMIT = None  # None for no limit
TOKEN_LIMIT = 50000
WORKERS = multiprocessing.cpu_count()
WINDOW = 5
DYNAMIC_WINDOW = False
NEGATIVE = 5  # 0 for plain hierarchical softmax (no negative sampling)
EPOCHES = 1

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

# def process_article((title, text)):
#     """
#     Parse a wikipedia article, returning its content as
#     `(title, list of tokens)`, all utf8.

#     """
#     text = gensim.corpora.wikicorpus.filter_wiki(text) # remove markup, get plain text
#     # tokenize plain text, throwing away sentence structure, short words etc
#     return title.encode('utf8'), gensim.utils.simple_preprocess(text)

# def convert_wiki(infile, processes=multiprocessing.cpu_count()):
#     """
#     Yield articles from a bz2 Wikipedia dump `infile` as (title, tokens) 2-tuples.

#     Only articles of sufficient length are returned (short articles & redirects
#     etc are ignored).

#     Uses multiple processes to speed up the parsing in parallel.

#     """
#     pool = multiprocessing.Pool(processes)
#     texts = gensim.corpora.wikicorpus._extract_pages(bz2.BZ2File(infile)) # generator
#     ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
#     # process the corpus in smaller chunks of docs, because multiprocessing.Pool
#     # is dumb and would try to load the entire dump into RAM...
#     for group in gensim.utils.chunkize(texts, chunksize=10 * processes):
#         for title, tokens in pool.imap(process_article, group):
#             if len(tokens) >= 50 and not any(title.startswith(ignore + ':') for ignore in ignore_namespaces):
#                 yield title.replace('t', ' '), tokens
#     pool.terminate()

# for title, tokens in covert_wiki('enwiki-latest-pages-articles.xml.bz2'):
#     print "%st%s" % (title, ' '.join(tokens))


if __name__ == '__main__':
    usage = 'python run_kernel_word2vec.py [corpus_path] [output_dir]'
    try:
        program = os.path.basename(sys.argv[0])
        corpus_path = sys.argv[1]
        output_dir = sys.argv[2]
    except:
        print usage
        sys.exit()

    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger(program)
    logger.info("running %s" % " ".join(sys.argv))

    outf = lambda prefix: os.path.join(output_dir, prefix)
    logger.info("output file template will be %s" % outf('PREFIX'))

    sentences = MyCorpus(corpus_path)

    if os.path.exists(outf('word2id')):
        logger.info("dictionary found, loading")
        word2id = utils.unpickle(outf('word2id'))
    else:
        logger.info("dictionary not found, creating")
        id2word = corpora.Dictionary(sentences, prune_at=10000000)
        id2word.filter_extremes(keep_n=TOKEN_LIMIT)  # filter out too freq/infreq words
        word2id = dict((v, k) for k, v in id2word.iteritems())
        utils.pickle(word2id, outf('word2id'))
    id2word = utils.revdict(word2id)

    # Filter all wiki documents to contain only those words.
    corpus = lambda: ([word for word in sentence if word in word2id] for sentence in sentences)

    if os.path.exists(outf('kw2v_%s' % GAMMA)):
        logger.info("Kernel word2vec model found, loading")
        # model = utils.unpickle(outf('kw2v'))
        model = Word2Vec.load_word2vec_format(outf('kw2v_%s' % GAMMA), binary=True)
    else:
        logger.info("Kernel word2vec model not found, creating")
        if NEGATIVE:
            model = Word2Vec(size=DIM, gamma=GAMMA, min_count=MIN_COUNT, window=WINDOW, workers=WORKERS, sg=1, negative=NEGATIVE, iter=EPOCHES)
        else:
            model = Word2Vec(size=DIM, gamma=GAMMA, min_count=MIN_COUNT, window=WINDOW, workers=WORKERS, sg=1, hs=1, negative=0, iter=EPOCHES)
        model.build_vocab(corpus())
        model.train(corpus())  # train with 1 epoch
        # model.build_vocab(sentences)
        # model.train(sentences)  # train with 1 epoch
        # trim unneeded model memory = use (much) less RAM
        # model.init_sims(replace=True)
        # model.word2id = dict((w, v.index) for w, v in model.vocab.iteritems())
        # model.id2word = utils.revdict(model.word2id)
        # model.word_vectors = model.syn0norm
        # utils.pickle(model, outf('kw2v'))

        print "total training time: %ss" % model.total_train_time
        model.save_word2vec_format(outf('kw2v_%s' % GAMMA), binary=True)
