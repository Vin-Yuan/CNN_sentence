import numpy as np
import re
import itertools
from collections import Counter
from collections import defaultdict
import pandas as pd
import os
import cPickle
import ipdb

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

class DataProcessor(object):
    def __init__(self, embedding_size=300, maxSentenceLength=100):
        self.train_text = None
        self.dev_text = None
        self.train_x = None
        self.train_y = None
        self.labels_name = None
        self.dev_x = None
        self.dev_y = None
        self.W_list = [] 
        self.W_words_maps = []
        self.embedding_size = embedding_size
        self.vocab = None
        self.maxSentenceLength = maxSentenceLength
    # load text and labels from train file:
    def load_train_file(self, filepath):
        self.train_text, labels = self.load_data_and_labels(filepath)
        self.train_text = [x.decode('utf8') for x in self.train_text]
        self.vocab, self.maxSentenceLength = self.getVocabulary(self.train_text)
        print 'vocab size', len(self.vocab)
        print 'average sentence length', self.maxSentenceLength
        self.train_y, self.labels_name = self.get_Y(labels)
    def load_dev_file(self, filepath):
        self.dev_text, labels = self.load_data_and_labels(filepath)
        self.dev_text = [x.decode('utf8') for x in self.dev_text]
        self.dev_y, _ = self.get_Y(labels)
    def load_data_and_labels(self, filepath):
        # expected csv file
        records = []
        with open(filepath, 'r') as f:
            records = f.readlines()
        labels = []
        sentences = []
        for line in records:
            piv = line.find('\t')
            labels.append(line[0:piv])
            sentences.append(line[piv+1:])
        # clean sentences
        #sentences = [clean_str(sent) for sent in sentences]
        return sentences, labels

    def get_Y(self, labels):
        """
            y: a list of list, which every item is a one-hot vector
        """
        if self.labels_name == None:
            class_count = Counter(labels)
            labels_name = class_count.keys()
            labels_name.sort()          # lexicographical
            class_num = len(labels_name)
            labels_map = {}
            for idx, name in enumerate(labels_name):
                temp = [0] * class_num
                temp[idx] = 1
                labels_map[name] = temp
            y = [labels_map[x] for x in labels]
            y = np.array(y, dtype=np.float32)
        else:
            labels_map = dict(enumerate(self.labels_name))
            labels_map = {v:k for k, v in labels_map.items()}
            y = np.zeros((len(labels), len(self.labels_name)), dtype=np.float32)
            for idx, label in enumerate(labels):
                y[idx][labels_map[label]] = 1
            labels_name = self.labels_name
        return y, labels_name
    
    def add_unknown_words(self, word_vecs, vocab, min_df=1, k=300):
        """
        For words that occur in at least min_df documents, create a separate word vector.    
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        """
        for word in vocab:
            if word not in word_vecs and vocab[word] >= min_df:
                word_vecs[word] = np.random.uniform(-0.25,0.25,k) 

    def getLookUpTable(self, word_vecs):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        embedding_size = word_vecs.values()[0].shape[-1]
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, embedding_size), dtype='float32')
        W[0] = np.zeros(embedding_size, dtype='float32')
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def add_W(self,word2vecFile=None, name="W"):
        if word2vecFile:
            fileDir, filename = os.path.split(word2vecFile)
            filename = os.path.splitext(filename)[0]
            filename = filename + '_' + name
            processed_w2v_file = os.path.join(fileDir,filename+'.select')
            if os.path.isfile(processed_w2v_file):
                temp = cPickle.load(open(processed_w2v_file, 'rb'))
                w2v_map = temp[0]
            else:
                w2v_map = load_bin_vec(word2vecFile, self.vocab)
                cPickle.dump([w2v_map], open(processed_w2v_file, 'wb'))
            # may be adjust the w2v_map to form same embedding_size
            embedding_size = w2v_map.values()[0].shape[-1]
            self.add_unknown_words(w2v_map, vocab=self.vocab, k=embedding_size)
            W, words_map = self.getLookUpTable(w2v_map) 
            self.W_list.append(W)
            self.W_words_maps.append(words_map)
        else:
            w2v_map = {}
            self.add_unknown_words(w2v_map, vocab=self.vocab, k=self.embedding_size)
            W, words_map = self.getLookUpTable(w2v_map)
            self.W_list.append(W)
            self.W_words_maps.append(words_map)
    def getOriginVocab(self):
        words = []
        maxSentenceLength = 0
        for line in self.train_text:
            temp = list(line) 
            maxSentenceLength = max(maxSentenceLength, len(temp))
            words.extend(temp)
        words_count = Counter(words)
        words_count = words_count.most_common(len(words_count))
        with open('words_statistic.txt', 'w') as f:
            for k, v in words_count:
                #f.writelines('{0} {1}\n'.format(k, v))
                f.writelines('{0} {1}\n'.format(k.encode('utf8'),v))
        print 'origin vocabulary size (statistic) : writing to words_statistic.txt', len(words_count)

    def getVocabulary(self, x_text, max_vocabulary_size=8000):
        words = []
        maxSentenceLength = 0
        for line in x_text:
            temp = list(line) 
            #maxSentenceLength = max(maxSentenceLength, len(temp))
            maxSentenceLength += len(temp)
            words.extend(temp)
        # get average nums of sentence length
        maxSentenceLength = maxSentenceLength / len(x_text) + 5
        words_count = Counter(words)
        # filter  words
        '''
        if len(words_count) > max_vocabulary_size:
            print 'origin vocabulary size (statistic) : ', len(words_count)
            words_count = words_count.most_common(max_vocabulary_size- 1)
            print 'filter vocabulary size (used) : ', len(words_count)
            words_count = dict(words_count)
        '''
        words_count = words_count.most_common(len(words_count))
        words_count = dict(words_count)
        return words_count, maxSentenceLength 

    def text2x(self, x_text):
        embedModel = zip(self.W_list, self.W_words_maps)
        collect_x = None
        for W, words_map in embedModel:
            x = np.zeros((len(x_text), self.maxSentenceLength))
            for i, line in enumerate(x_text):
                #words = line.split(' ')
                words = list(line)
                end = min(self.maxSentenceLength, len(words))
                for j in range(end):
                    if(words_map.has_key(words[j])):
                        x[i][j] = words_map[words[j]]
                    else:
                        x[i][j] = 0
            x = np.expand_dims(x, axis=-1)
            if collect_x is None:
                collect_x = x
            else:
                collect_x = np.concatenate((collect_x, x), axis=-1)
        return collect_x
    
    def train_text2x(self):
        x = self.text2x(self.train_text)
        self.train_x = np.array(x, dtype=np.float32)
    
    def dev_text2x(self):
        x = self.text2x(self.dev_text)
        self.dev_x = np.array(x, dtype=np.float32)
    # batch_iter on processed data
    def batch_iter(self, x, y,  batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = list(zip(x, y))
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(data_size/batch_size) + 1
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

if __name__ == "__main__":
    train_file = 'data/backup/TREC/train.label'
    dev_file = 'data/backup/TREC/dev.label'
    #x_text, y = load_data_and_labels_(filename)
    data_processor = DataProcessor()
    data_processor.load_train_file(train_file)
    data_processor.load_dev_file(dev_file)
    data_processor.add_W('./data/GoogleNews-vectors-negative300.bin', name='TREC',)
    data_processor.add_W(name='Random')
    data_processor.train_text2x()
    data_processor.dev_text2x()
    print 'vocab size', len(data_processor.vocab)
    print 'W:',data_processor.W_list[0].shape
    print 'x:',data_processor.train_x[:,:,0].shape
    print 'label:',data_processor.labels_name
    print 'train:', data_processor.train_x.shape
    print 'dev:', data_processor.dev_x.shape
    '''
    for batch in data_processor.batch_iter(data_processor.train_x,data_processor.train_y,
        batch_size=512, num_epochs=1, shuffle=True):
        print batch.shape
    '''
