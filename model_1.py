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
y_in = Input(shape=(None,)) # 实体标记
t_in = Input(shape=(1,)) # 是否有关联（标签）

x1, x2, y, t = x1_in, x2_in, y_in, t_in
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


train_model = Model([x1_in, x2_in, y_in, t_in],
                    [pt])


pt_loss = K.mean(K.binary_crossentropy(t, pt))

loss =pt_loss

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
            X1, X2,Y, T = [], [], [], []
            for i in idxs:
                try:
                    text=self.data[i][0]
                    x1 = [char2id.get(c, 1) for c in text]
                    x2 = self.data[i][4]
                    x2 = [char2id.get(c, 1) for c in x2]
                    X1.append(x1)
                    X2.append(x2)
                    Y.append(self.data[i][2])
                    T.append(self.data[i][3])
                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = seq_padding(X1)
                        X2 = seq_padding(X2)
                        Y = seq_padding(Y)
                        T = seq_padding(T)
                        yield [X1, X2,  Y, T], None
                        X1, X2, Y, T = [], [], [], []
                except:
                    pass







train_D = data_generator(train_data)

train_model.fit_generator(train_D.__iter__(),
                          steps_per_epoch=len(train_D),
                          epochs=1,
                          # callbacks=[evaluator]
                         )
train_model.save_weights('./data/1_best_model.weights')





