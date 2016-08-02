#-*-coding:utf-8 -*-
from flask import Flask, jsonify, Response, request, json
import numpy as np
import os
import sys
from data_helpers import DataProcessor
from text_cnn import TextCNN
import tensorflow as tf
from tensorflow import app
import ipdb

#reload(sys)
#sys.setdefaultencoding('utf-8')
app = Flask(__name__)

labels_map = {
   '1':u"续保",
   '2':u"增加险种",
   '3':u"删除险种",
   '4':u"修改险种",
   '5':u"保险相关的通用问答",
   '6':u"保单金额增加",
   '7':u"保单金额减少",
   '8':u"不满意",
   '9':u"满意",
   '10':u"确定",
   '11':u"取消",
   '12':u"疑问",
   '13':u"保险无关的闲聊",
   '14':u"保单金额查询",
   '15':u"保单险种列表",
   '16':u"已有具体险种查询",
   '17':u"已删具体险种查询",
   '18':u"转人工服务"
}


@app.route("/")
def hello():
    return "Hello World!"

def getCharsMap(embedding_file):
    data_processor = DataProcessor()
    _, W_words_maps = data_processor.loadWordsMap(input_file=embedding_file)
    words_map = W_words_maps[0]
    return words_map 
def sentence2index(sentence, chars_map, maxSentenceLength):
    if isinstance(sentence, str):
        sentence = sentence.decode('utf8')
    char_list = list(sentence)
    vector = np.zeros(maxSentenceLength,dtype=np.float32)
    length = min(maxSentenceLength, len(char_list))
    for i in range(length):
        char = char_list[i]
        if(chars_map.has_key(char)):
            vector[i] = chars_map[char]
    temp = np.expand_dims(vector, axis=0)
    return np.expand_dims(temp, axis=-1)

def init_classifier(checkpoint_dir):
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.get_default_graph()
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    sess = tf.Session(config=session_conf)
    print 'load session and graph'
    with graph.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
    return graph, sess
# get the label and score 
def label_score(text):
    #Argument:
    #   text: the text to classify
    #Return:
    #   confidence and label index
    input_x = app.graph.get_operation_by_name("input_x").outputs[0]
    dropout_keep_prob = app.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
    predictions = app.graph.get_operation_by_name("output/predictions").outputs[0]
    scores = app.graph.get_operation_by_name("output/scores").outputs[0]
    text = [text]
    x = app.data_processor.text2x(text)
    class_num = scores.get_shape().as_list()[-1]
    print "class num is : {}".format(class_num)
    top_k = 5
    if(class_num < top_k):
        top_k = class_num
    softmax = tf.nn.softmax(scores, name="softmax")
    values, indices = tf.nn.top_k(softmax, k=top_k)
    label = []
    confidence = []
    label, confidence = app.sess.run([indices, values],{input_x: x, dropout_keep_prob: 1.0})
    return label, confidence 
 
def one_test():
    app.CHAR_MAP = getCharsMap('data/insure3/embedding.cpkl')
    test_sentence = "保险"
    labelIdx, confidence = label_score(test_sentence)
    labelIdx = labelIdx[0]
    confidence = confidence[0]
    for idx, label in enumerate(labelIdx):
        print labels_name[label], confidence[idx]

@app.route('/api/test/<text>') 
def test(text):
    if request.method == 'GET':
        key = ['a','b','c']
        val = [(1,0.9),(2,0.8),(3,0.5)]
        info = [
            {'key':key},
            {'val':val}
        ]
        result = {text:1}
        return json.dumps(info, ensure_ascii=False)
        #return jsonify(results=info)

@app.route('/api/inference/<text>')
def inference(text):
    error = None
    if request.method == 'GET':
        labelIdx, confidence = label_score(text)
        labelIdx = labelIdx[0]
        confidence = confidence[0]
        result = {}

        for i, idx in enumerate(labelIdx):
            label = app.data_processor.labels_name[idx]
            name = labels_map[label] 
            #probability = float('%0.3f'%(confidence[i]))
            probability = float(confidence[i])
            result[name] = (label, probability)
        #return jsonify(results=result, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

if __name__ == "__main__":
    log_dir = 'data/insure'
    app.data_processor = DataProcessor()
    app.data_processor.restore(os.path.join(log_dir, 'data_processor.cpkl'))
    app.graph, app.sess = init_classifier(os.path.join(log_dir,'runs/checkpoints'))

    app.run(
        host="0.0.0.0",
        debug=True
    )
