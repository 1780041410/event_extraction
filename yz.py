import json
from tqdm import tqdm
import os
import numpy as np
from random import choice
from itertools import groupby
# class data_generator(object):
#    def __init__(self,value):
#       self.value = value
#    def __len__(self):
#       return self.value
#
#
#    def __iter__(self):
#       while True:
#          for i in range(10):
#             if self.value==1:
#                yield [1,2,3,4,5,6,7],None
#                x1,x2,x3,x4,x5,x6=[],[],[],[],[],[]
# s=data_generator(1)
#
# for i in s.__iter__():
#    print(i)

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

import json
from tqdm import tqdm
import os
import numpy as np
from random import choice
from itertools import groupby
'''
{
"text_id":"1",
"text":"比特币吸粉无数，但央行的心另有所属|界面新闻 · jmedia"
"mention_data":[
        {
            "kb_id":"278410",
            "mention":"比特币",
            "offset":"0"
        },
        {
            "kb_id":"199602",
            "mention":"央行",
            "offset":"9"
        },
        {
            "kb_id":"215472",
            "mention":"界面新闻",
            "offset":"18"
        }
    ]
}
'''

# a=[('电影', 5, '336024'), ('导演', 15, '117574')]
#
# with open('./data/ccks2019_el/develop.json') as f:
#     for line in tqdm(f):
#         _ = json.loads(line)
#         id,text=_['text_id'],_['text']
#         print(id)
# class Dynamic_connection(Layer):
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0][0], input_shape[0][1],input_shape[0][2])
#
#     def build(self, input_shape):
#         self.x1_length = input_shape[0][1]
#         self.x1_hidden_size = input_shape[0][2]
#         self.x2_length=input_shape[1][1]
#         self.x2_hidden_size=input_shape[1][2]
#
#     def call(self, inputs,mask=None):
#         x1,x2,begin,end=inputs
#         x2_embedding=tf.reshape(x2,shape=[-1,self.x2_hidden_size])
#         begin = tf.cast(begin, dtype=tf.int32)
#         begin=begin+tf.cast(tf.reshape(tf.range(0,tf.shape(begin)[0]*tf.shape(x2)[-1],tf.shape(x2)[-1]),shape=[tf.shape(begin)[0],1]),dtype=tf.int32)
#         x_b=tf.nn.embedding_lookup(x2_embedding, begin)
#         x_b=tf.reshape(x_b,shape=[-1,self.x1_length,self.x2_hidden_size])
#
#         end = tf.cast(end, dtype=tf.int32)
#         end = end + tf.cast(tf.reshape(tf.range(0, tf.shape(end)[0] * tf.shape(x2)[-1], tf.shape(x2)[-1]),shape=[tf.shape(end)[0], 1]), dtype=tf.int32)
#         x_e = tf.nn.embedding_lookup(x2_embedding, end)
#         x_e = tf.reshape(x_e, shape=[-1, self.x1_length, self.x2_hidden_size])
#         x=concatenate([x_b,x_e],axis=-1)
#         # out=concatenate([x_b,x_e],axis=-1)
#         out=tf.add(x,x1)
#         return out

import numpy

# def find_lcseque(s1, s2):
#      # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
#     m = [ [ 0 for x in range(len(s2)+1) ] for y in range(len(s1)+1) ]
#     # d用来记录转移方向
#     d = [ [ None for x in range(len(s2)+1) ] for y in range(len(s1)+1) ]
#
#     for p1 in range(len(s1)):
#         for p2 in range(len(s2)):
#             if s1[p1] == s2[p2]:            #字符匹配成功，则该位置的值为左上方的值加1
#                 m[p1+1][p2+1] = m[p1][p2]+1
#                 d[p1+1][p2+1] = 'ok'
#             elif m[p1+1][p2] > m[p1][p2+1]:  #左值大于上值，则该位置的值为左值，并标记回溯时的方向
#                 m[p1+1][p2+1] = m[p1+1][p2]
#                 d[p1+1][p2+1] = 'left'
#             else:                           #上值大于左值，则该位置的值为上值，并标记方向up
#                 m[p1+1][p2+1] = m[p1][p2+1]
#                 d[p1+1][p2+1] = 'up'
#     (p1, p2) = (len(s1), len(s2))
#     s = []
#     while m[p1][p2]:    #不为None时
#         c = d[p1][p2]
#         if c == 'ok':   #匹配成功，插入该字符，并向左上角找下一个
#             s.append(s1[p1-1])
#             p1-=1
#             p2-=1
#         if c =='left':  #根据标记，向左找下一个
#             p2 -= 1
#         if c == 'up':   #根据标记，向上找下一个
#             p1 -= 1
#     s.reverse()
#     return ''.join(s)
# import pickle
# with open('./data/kb.pkl','rb') as f:
#     id2kb = pickle.load(f)
#     kb2id=pickle.load(f)
#     id2char = pickle.load(f)
#     char2id = pickle.load(f)
# from operator import itemgetter
# match_data_dic={}
# text='南京南站'
#
# for  k,v in kb2id.items():
#     match_length=len(find_lcseque('瑞金',k))
#     match_data_dic[' '.join(v)]=match_length
# match_data_dic=sorted(match_data_dic.items(),key=lambda item:item[1],reverse=True)
# # print(match_data_dic['南京南站'])
# i=0
# id_list=[]
# for k in match_data_dic:
#     print(k)
#     id_list.append(k[0])
#     i+=1
#     if i>=4:
#         break
# id_list=list(set(id_list))
# print(id_list)
# with open('./result.txt') as fr:
#     for line in fr.readlines():
#         a,b=line.split('\t')
#         print(type(a),type(b))
#         break
#
# print(type(a))