import pandas as pd
import pickle
def writePKL(data, file):
    with open(file, 'wb') as fw:
        pickle.dump(data, fw)
def readPKL(file):
    with open(file, 'rb') as fr:
        return pickle.load(fr)
labels=['资金账户风险', '其他', '涉嫌非法集资', '涉嫌传销', '资产负面', '产品违规', '失联跑路', '歇业停业', '涉嫌违法', '投诉维权', '涉嫌欺诈', '公司股市异常', '不能履职', '评级调整', '高管负面', '提现困难', '交易违规', '业绩下滑', '实控人股东变更', '重组失败', '财务造假', '信批违规']
labels={ j:i for i,j in enumerate(labels)}
char2id=readPKL("./data/id2char.pkl")
id2char = readPKL("./data/char2id.pkl")

def getData(train_path,test_path):
    df_train=pd.read_csv(train_path,names=['id','text','type','entity'],header=None)
    df_test=pd.read_csv(test_path,names=['id','text','type','entity'],header=None)
    df_data=pd.concat([df_train,df_test],axis=0)
    print(len(df_train))
    print(len(df_data))
    return df_train
def getCharIndex(df_data):
    texts = list(df_data['text'].values)
    char2id = {}
    id2char = {}
    chars = {}
    i = 0
    for text in texts:
        try:
            i += 1
            for c in text:
                chars[c] = chars.get(c, 0) + 1
        except:
            print(i, text)
    chars = {i: j for i, j in chars.items() if j >= 2}
    char2id = {i + 2: j for i, j in enumerate(chars)}
    id2char = {j: i for i, j in char2id.items()}
    writePKL(char2id, './data/char2id.pkl')
    writePKL(id2char, './data/id2char.pkl')

def convertText2id(text):
    ids=[char2id.get(c,'1')for c in text]
    return ' '.join([str(i)for i in ids])
def type2id(type):
    vec=len(labels)*['0']
    vec[labels.get(type)]='1'
    return ' '.join(vec)
def begin_entity(text,entity):
    vec = ['0'] * len(text)
    try:
        index=text.find(entity)
        vec[index]='1'
    except:
        return ' '.join(vec)
    return ' '.join(vec)
def end_entity(text,entity):
    vec = ['0'] * len(text)
    try:
        index = text.find(entity)
        vec[index+len(entity)-1] = '1'
    except:
        return ' '.join(vec)
    return ' '.join(vec)

# df_data=getData('./data/event_type_entity_extract_train.csv','./data/event_type_entity_extract_eval.csv')
#
# df_data['type2id']=df_data['type'].apply(type2id)
# df_data['text_id']=df_data['text'].apply(convertText2id)
# df_data['s1'] = df_data.apply(lambda row: begin_entity(row['text'], row['entity']), axis=1)
# df_data['s2'] = df_data.apply(lambda row: end_entity(row['text'], row['entity']), axis=1)
# df_data.to_csv('./data/result.csv',index = False)

#
#
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
def seq_and_vec(x):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq, vec = x
    vec = K.expand_dims(vec, 1)
    vec = K.zeros_like(seq[:, :, :1]) + vec
    return K.concatenate([seq, vec], 2)

x1_in = Input(shape=(None,)) #句子输入
s1_in = Input(shape=(None,)) # 实体左边界（标签）
s2_in = Input(shape=(None,)) # 实体右边界（标签）
y_in = Input(shape=(22,)) # 实体标签--->类型

x1,s1, s2,y= x1_in,s1_in, s2_in, y_in
x1_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)
embedding=Embedding(len(id2char)+2,char_size)
x1 = embedding(x1)
x1 = Dropout(0.2)(x1)
a_dim=K.int_shape(x1)[-1]+K.int_shape(y)[-1]
x1=Lambda(seq_and_vec,output_shape=(None,a_dim))([x1,y])
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])
x1 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x1)
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])
x1 = Bidirectional(CuDNNLSTM(char_size//2, return_sequences=True))(x1)
x1 = Lambda(lambda x: x[0] * x[1])([x1, x1_mask])
x_max = Lambda(seq_maxpool)([x1, x1_mask])
t_dim = K.int_shape(x1)[-1]
h = Lambda(seq_and_vec, output_shape=(None, t_dim*2))([x1, x_max])
h = Conv1D(char_size, 3, activation='relu', padding='same')(h)
ps1 = Dense(1, activation='sigmoid')(h)
ps2 = Dense(1, activation='sigmoid')(h)
#shape=(?, ?, 1)
s_model=Model([x1_in,y_in],[ps1,ps2])
train_model = Model([x1_in,s1_in, s2_in,y_in],
                    [ps1, ps2])
s1 = K.expand_dims(s1, 2)
s2 = K.expand_dims(s2, 2)

s1_loss = K.binary_crossentropy(s1, ps1)
s1_loss = K.sum(s1_loss * x1_mask) / K.sum(x1_mask)
s2_loss = K.binary_crossentropy(s2, ps2)
s2_loss = K.sum(s2_loss * x1_mask) / K.sum(x1_mask)

loss = s1_loss + s2_loss

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(1e-3))
train_model.summary()

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

# train_data=readPKL('./e_train.pkl')
# df_train=pd.read_csv('./data/result.csv')

# print(df_train.head())
# for i in range(0, len(df_train)):
#     train_data.append([df_train.iloc[i]['text'],[ int(i)for i in df_train.iloc[i]['s1'].split()],[ int(i) for i in df_train.iloc[i]['s2'].split()],[int(i)for i in df_train.iloc[i]['type2id'].split()]])
# writePKL(train_data,'./e_train.pkl')

#

# random_order=range(len(train_data))
# np.random.permutation(random_order)
# dev_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 == 0]
# train_data = [train_data[j] for i, j in enumerate(random_order) if i % 9 != 0]



class data_generator:
    def __init__(self, data, batch_size=64):
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
            X1,S1, S2, Y = [], [], [], []
            for i in idxs:
                text = self.data[i][0]
                x1 =[char2id.get(c, 1) for c in text]
                s1, s2 = self.data[i][1],self.data[i][2]
                y = self.data[i][-1]
                X1.append(x1)
                S1.append(s1)
                S2.append(s2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    S1 = seq_padding(S1)
                    S2 = seq_padding(S2)
                    Y = seq_padding(Y)
                    yield [X1,S1, S2, Y], None
                    X1,S1, S2, Y = [], [], [], []
# evaluator = Evaluate()
# train_D = data_generator(train_data)
# train_model.fit_generator(train_D.__iter__(),
#                           steps_per_epoch=len(train_D),
#                           epochs=10
#                           # callbacks=[evaluator]
#
#                           )
# train_model.save_weights('./event_best_model.weights')
#
#





train_model.load_weights('./event_best_model.weights')





df_test = pd.read_csv('./data/event_type_entity_extract_eval.csv', names=['id', 'text', 'type', 'entity'], header=None)

def result_test(text_in,type):
    y = len(labels) * [0]
    y[labels.get(type)] = 1
    x1 = [char2id.get(c, 1) for c in text_in]
    x1 = np.array([x1])
    y=np.array([[int(i) for i in y]])
    _k1, _k2 = s_model.predict([x1,y])
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    _k1, _k2 = np.where(_k1 > 0.5)[0], np.where(_k2 > 0.5)[0]
    _subjects = []
    for i in _k1:
        j = _k2[_k2 >= i]
        if len(j) > 0:
            j = j[0]
            _subject = text_in[i: j + 1]
            _subjects.append((_subject, i, j))

    return  _subjects
result=[]
for i in range(0, len(df_test)):
    id=df_test.iloc[i]['id']
    text=df_test.iloc[i]['text']
    type=df_test.iloc[i]['type']
    if id==102213 or type=='其他':
        result.append([id,''])
    else:
        t=result_test(text,type)
        if t==[]:
            result.append([id,''])
        else:
            result.append([id,t[0][0]])
from tqdm import tqdm
df_data=pd.DataFrame()
print(result)

for i in tqdm(result):
    item={}
    item['x1']=str(int(i[0]))
    item['x2']=i[1]
    df_data = df_data.append(item, ignore_index=True)
df_data.to_csv('./result.txt',index=False,header=None,sep='\t')
# with open('./result.txt','w',encoding='utf-8') as fw:
#     for i in tqdm(result):
#         fw.write(str(i[0])+'\t'+str(i[1])+'\n')



















# text_in='同大股份(300321)股东减持210.6万股 占比4.74%大通燃气控股权拟溢价转让 实控人将变更,实控人股东变更'

#
# x1 = [char2id.get(c, 1) for c in text_in]
# x1 = np.array([x1])
# print(x1)
# y='0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0'.split()
# y=np.array([[int(i) for i in y]])
#
# _k1, _k2 = s_model.predict([x1,y])
# _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
# _k1, _k2 = np.where(_k1 > 0.5)[0], np.where(_k2 > 0.5)[0]
# _subjects = []
# print(_k1, _k2)
# for i in _k1:
#     j = _k2[_k2 >= i]
#     if len(j) > 0:
#         j = j[0]
#         _subject = text_in[i: j + 1]
#         _subjects.append((_subject, i, j))
# print(_subjects)