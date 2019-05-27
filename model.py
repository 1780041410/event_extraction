import json
from tqdm import tqdm
import os
import numpy as np
from random import choice
from itertools import groupby
import pickle
mode = 0
min_count = 2
char_size = 128



with open('./data/kb.pkl','rb') as f:
    id2kb = pickle.load(f)
    kb2id=pickle.load(f)
    id2char = pickle.load(f)
    char2id = pickle.load(f)
with open('./data/train.pkl','rb') as f:
    all_alies = pickle.load(f)
train_data=all_alies
random_order=range(len(train_data))
np.random.permutation(random_order)


train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != 0]

for i in train_data[:10]:
    print(i)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


char_size=128
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq,mask=x
    seq-=(1-mask)*1e10
    return K.max(seq,1)
x1_in = Input(shape=(None,)) # 待识别句子输入
x2_in = Input(shape=(None,)) # 实体语义表达输入
s1_in = Input(shape=(None,)) # 实体左边界（标签）
s2_in = Input(shape=(None,)) # 实体右边界（标签）
y_in = Input(shape=(None,)) # 实体标记
t_in = Input(shape=(1,)) # 是否有关联（标签）

x1, x2, s1, s2, y, t = x1_in, x2_in, s1_in, s2_in, y_in, t_in
x1_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)
x2_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x2)
embedding=Embedding(len(id2char)+2,char_size)
x1 = embedding(x1)
x1 = Dropout(0.2)(x1)
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])
x1 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x1)
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])
x1 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x1)
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])

h = Conv1D(char_size, 3, activation='relu', padding='same')(x1)
ps1 = Dense(1, activation='sigmoid')(h)
ps2 = Dense(1, activation='sigmoid')(h)
#shape=(?, ?, 1)
s_model=Model(x1_in,[ps1,ps2])

y=Lambda(lambda x:K.expand_dims(x,2))(y)
x1=Concatenate()([x1,y])
x1=Conv1D(char_size,3,padding='same')(x1)


x2 = embedding(x2)
x2 = Dropout(0.2)(x2)
x2 = Lambda(lambda x: x[0] * x[1])([x2, x2_mask])
x2 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x2)
x2 = Lambda(lambda x: x[0] * x[1])([x2, x2_mask])
x2 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x2)
x2 = Lambda(lambda x: x[0] * x[1])([x2, x2_mask])

x1=Lambda(seq_maxpool)([x1,x1_mask])
x2=Lambda(seq_maxpool)([x2,x2_mask])
x12=Multiply()([x1,x2])
x=Concatenate()([x1,x2,x12])

x = Dense(char_size, activation='relu')(x)
pt = Dense(1, activation='sigmoid')(x)
# shape=(?, 1)
t_model = Model([x1_in, x2_in, y_in], pt)


train_model = Model([x1_in, x2_in, s1_in, s2_in, y_in, t_in],
                    [ps1, ps2, pt])

s1 = K.expand_dims(s1, 2)
s2 = K.expand_dims(s2, 2)

s1_loss = K.binary_crossentropy(s1, ps1)
s1_loss = K.sum(s1_loss * x1_mask) / K.sum(x1_mask)
s2_loss = K.binary_crossentropy(s2, ps2)
s2_loss = K.sum(s2_loss * x1_mask) / K.sum(x1_mask)
pt_loss = K.mean(K.binary_crossentropy(t, pt))

loss = s1_loss + s2_loss + pt_loss

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(1e-3))
train_model.summary()




class data_generator:
    def __init__(self, data, batch_size=128):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            X1, X2, S1, S2, Y, T = [], [], [], [], [], []
            for i in idxs:
                try:
                    text=self.data[i][0]
                    x1 = [char2id.get(c, 1) for c in text]
                    s1, s2 = np.array(self.data[i][-2]),np.array(self.data[i][-1])
                    x2 = self.data[i][4]
                    x2 = [char2id.get(c, 1) for c in x2]
                    X1.append(x1)
                    X2.append(x2)
                    S1.append(s1)
                    S2.append(s2)
                    Y.append(self.data[i][2])
                    T.append(self.data[i][3])
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        Y = seq_padding(Y)
                        T = seq_padding(T)
                        yield [X1, X2, S1, S2, Y, T], None
                        X1, X2, S1, S2, Y, T = [], [], [], [], [], []
                except:
                    pass


def find_lcseque(s1, s2):
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    # d用来记录转移方向
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    s = []
    while m[p1][p2]:  # 不为None时
        c = d[p1][p2]
        if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  # 根据标记，向左找下一个
            p2 -= 1
        if c == 'up':  # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    return ''.join(s)
def match_id(s):
    match_data_dic = {}
    for k, v in kb2id.items():
        match_length = len(find_lcseque(s, k))
        match_data_dic[' '.join(v)] = match_length
    match_data_dic = sorted(match_data_dic.items(), key=lambda item: item[1], reverse=True)
    i = 0
    id_list = []
    for k in match_data_dic:
        id_list.extend(k[0].split(' '))
        i += 1
        if i >=4:
            break
    id_list = list(set(id_list))
    return id_list
def extract_items(text_in):
    _x1 = [char2id.get(c, 1) for c in text_in]
    _x1 = np.array([_x1])
    _k1, _k2 = s_model.predict(_x1)
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    _k1, _k2 = np.where(_k1 > 0.5)[0], np.where(_k2 > 0.5)[0]
    _subjects = []
    for i in _k1:
        j = _k2[_k2 >= i]
        if len(j) > 0:
            j = j[0]
            _subject = text_in[i: j+1]
            _subjects.append((_subject, i, j))
    print(_subjects)
    if _subjects:
        R = []
        _X2, _Y = [], []
        _S, _IDXS = [], {}
        for _s in _subjects:
            _y = np.zeros(len(text_in))
            _y[_s[1]: _s[2]] = 1
            _IDXS[_s] = kb2id.get(_s[0], match_id(_s[0]))
            for i in _IDXS[_s]:
                _x2 = id2kb[i]['subject_desc']
                _x2 = [char2id.get(c, 1) for c in _x2]
                _X2.append(_x2)
                _Y.append(_y)
                _S.append(_s)
        if _X2:
            _X2 = seq_padding(_X2)
            _Y = seq_padding(_Y)
            _X1 = np.repeat(_x1, len(_X2), 0)
            scores = t_model.predict([_X1, _X2, _Y])[:, 0]
            for k, v in groupby(zip(_S, scores), key=lambda s: s[0]):
                v = np.array([j[1] for j in v])
                kbid = _IDXS[k][np.argmax(v)]
                R.append((k[0], k[1], kbid))
        return R
    else:
        return []


# class Evaluate(Callback):
#     def __init__(self):
#         self.F1 = []
#         self.best = 0.
#     def on_epoch_end(self, epoch, logs=None):
#         f1, precision, recall = self.evaluate()
#         self.F1.append(f1)
#         if f1 > self.best:
#             self.best = f1
#             train_model.save_weights('best_model.weights')
#         print ('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))
#     def evaluate(self):
#         A, B, C = 1e-10, 1e-10, 1e-10
#         for d in tqdm(iter(dev_data)):
#             R = set(extract_items(d['text']))
#             T = set(d['mention_data'])
#             A += len(R & T)
#             B += len(R)
#             C += len(T)
#         return 2 * A / (B + C), A / B, A / C


# evaluator = Evaluate()
train_D = data_generator(train_data)

train_model.fit_generator(train_D.__iter__(),
                          steps_per_epoch=len(train_D),
                          epochs=1,
                          # callbacks=[evaluator]
                         )
train_model.save_weights('./data/best_model.weights')

s_model.load_weights('./best_model.weights')
# text_in='《忏悔录》 20111016 荒唐出轨'

# print(extract_items(text_in))
# l=[]
# with open('./data/ccks2019_el/develop.json') as f,open('./test.json','w') as fw:
#     for line in tqdm(f):
#         _ = json.loads(line)
#         id,text=_['text_id'],_['text']
#         print(id)
#         a=extract_items(text)
#         if not a:
#             l.append(json.dumps({'text_id': id, 'text': text, 'mention_data':[{'kb_id':'NIL','mention':text,'offset':'0'}]}, ensure_ascii=False))
#         else:
#             l.append(json.dumps({'text_id':id,'text':text,'mention_data':[{'kb_id':str(i[2]),'mention':i[0],'offset':str(i[1])} for i in a ]}, ensure_ascii=False))
#     for c in l:
#         fw.write(c + '\n')