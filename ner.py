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
s1_in = Input(shape=(None,)) # 实体左边界（标签）
s2_in = Input(shape=(None,)) # 实体右边界（标签）
x1,s1, s2= x1_in,s1_in,s2_in
x1_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)
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
s_model=Model([x1_in,s1_in,s2_in],[ps1,ps2])


s1 = K.expand_dims(s1, 2)
s2 = K.expand_dims(s2, 2)

s1_loss = K.binary_crossentropy(s1, ps1)
s1_loss = K.sum(s1_loss * x1_mask) / K.sum(x1_mask)
s2_loss = K.binary_crossentropy(s2, ps2)
s2_loss = K.sum(s2_loss * x1_mask) / K.sum(x1_mask)

loss = s1_loss + s2_loss
s_model.add_loss(loss)
s_model.compile(optimizer=Adam(1e-3))
s_model.summary()



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
            X1,S1, S2 = [], [], []
            for i in idxs:
                try:
                    text=self.data[i][0]
                    x1 = [char2id.get(c, 1) for c in text]
                    s1, s2 = np.array(self.data[i][-2]),np.array(self.data[i][-1])
                    X1.append(x1)
                    S1.append(s1)
                    S2.append(s2)
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        yield [X1,S1,S2], None
                        X1,S1,S2 = [], [], []
                except:
                    pass




train_D = data_generator(train_data)

# for i in train_D.__iter__():
#     print(i)




# s_model.fit_generator(train_D.__iter__(),
#                           steps_per_epoch=len(train_D),
#                           epochs=2
#                          )
s_model.load_weights('best_model.weights')
text_in='《谢文东2》电视剧_全集(1-28集)高清在线观看'
_x1 = [char2id.get(c, 1) for c in text_in]
_x1 = np.array([_x1])
s1_1=np.array([0]*len(_x1))
s2_2=np.array([0]*len(_x1))
_k1, _k2 = s_model.predict([_x1,s1_1,s2_2])
_k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
_k1, _k2 = np.where(_k1 >=0.5)[0], np.where(_k2 >=0.5)[0]
_subjects = []
for i in _k1:
    j = _k2[_k2 >= i]
    if len(j) > 0:
        j = j[0]
        _subject = text_in[i: j + 1]
        _subjects.append((_subject, i, j))
print(_subjects)