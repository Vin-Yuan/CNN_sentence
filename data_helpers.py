import numpy as np
import re
import itertools
from collections import Counter
import cPickle
from collections import defaultdict
import pandas as pd
def build_data_cv(text,label, cv=10, clean_string=True):
    """ Loads data and split into 10 folds.
    Argument:
        text: a list which every item is a string of sentence
        label: every sentence's label, so len(text) == len(label)
    Return:
        sentences_info: a list that every item is a dict ['label','sentence','words_num','split']
        vocab: vocabulary statisitc info, a dict <word>:count 
    """
    sentences_info = []
    vocab = defaultdict(float)
    if clean_string:
        text = [clean_str(sentence) for sentence in text]
    else:
        text = [sentence.lower() for sentence in text]

    for idx, sentence in enumerate(text):
        sentence = sentence.strip()
        words = set(sentence.split(' '))
        for word in words:
            vocab[word] += 1
        datum = {
            "label": label[idx],
            "sentence": sentence,
            "words_num": len(sentence.split(' ')),
            "split": np.random.randint(0,cv)
        }
        sentences_info.append(datum)
    return sentences_info, vocab

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec and
    filter select those w2v which word in vocab
    --------------------------------------------------
    Argument:
        fname: the pre-trained word2vec model
        vocab: the vocabulary , dict type {'word':count}
    Return:
        word_vecs: dict for {'word':vector} 
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k) 
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # pos/neg label is 0 or 1
    pos_label = np.zeros(len(positive_examples))
    neg_label = np.ones(len(negative_examples))
    labels = np.concatenate((pos_label, neg_label))
    # Generate labels is [0,1] or [1,0]
    positive_labels = [[0,1] for _ in positive_examples]
    negative_labels = [[1,0] for _ in negative_examples] 
    y = np.concatenate([positive_labels, negative_labels], 0)
    build_word2vec_vocabulary('data/',x_text, labels,'data/word2vec.cpkl')
    
    return x_text, y

def getEmbedding(x_text, sequence_length, static=True, name=None):
    """
    Argument:
        x_text : [batchsize, sentence]
        sequence_length: the max length over all sentences, used for word2vec_map 
    Return:
        VocabEmbedding: [vocabulary_size, embedding_size], the vocabulary embedding as lookup tabel, a dict like 
            {name:str, embedding:np.narray, static:True(or False)}
        word2vec_map: [batchsize, sequence_length] ,the index which map word to vector
    """
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word2vec_map, vocab = x[0], x[1], x[2], x[3], x[4]
    VocabEmbedding = np.zeros((len(x_text), sequence_length))
    for idx, sentence in enumerate(x_text):
        words = sentence.split(' ')
        for j, word in enumerate(words):
            VocabEmbedding [idx][j] = word2vec_map[word]
    return VocabEmbedding, word2vec_map
def get_Y(label, class_num):
    """
     Argument:
        label: a list contains the label number
     Return:
        y: the np.narray [len(label), class_num], which every item is a one-hot vector
    """
    y = np.zeros((len(label), class_num))
    for idx, class_map in label:
        y[idx][class_map] = 1;
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def build_word2vec_vocabulary(w2v_file, text, label, dumpFilePath):
    """
    Argument:
        w2v_file: the binary file of pretrained w2v_file
        text: list of setences
        label: a list, label of every sentence in text
    Return:
        None
        this will dump a cPickle file which contain:
        sentences_info, vocab_w2v, random_w2v, word_idx_map, vocab
    """
    sentences_info, vocab = build_data_cv(text, label, cv=10, clean_string=True)
    maxSentenceLength = np.max(pd.DataFrame(sentences_info)["words_num"])
    print "number of sentences: " + str(len(sentences_info))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(maxSentenceLength)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    vocab_w2c, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    random_w2c, _ = get_W(rand_vecs) 
    cPickle.dump([sentences_info, vocab_w2v, random_w2v, word_idx_map, vocab], open(dumpFilePath, "wb"))
    print "dataset created!"
    
