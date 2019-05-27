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


def process_kb_data():
    id2kb = {}
    with open('./data/ccks2019_el/kb_data') as f:
        for l in tqdm(f):
            _ = json.loads(l)
            subject_id = _['subject_id']
            subject_alias = list(set([_['subject']] + _.get('alias', [])))
            subject_alias = [alias.lower() for alias in subject_alias]
            subject_desc = '\n'.join(u'%s：%s' % (i['predicate'], i['object']) for i in _['data'])
            subject_desc = subject_desc.lower()
            if subject_desc:
                id2kb[subject_id] = {'subject_alias': subject_alias, 'subject_desc': subject_desc}

    kb2id = {}
    for i,j in id2kb.items():
        for k in j['subject_alias']:
            if k not in kb2id:
                kb2id[k] = []
            kb2id[k].append(i)
    #
    for i,j in id2kb.items():
        if i=="311223":
            print(i,j)
    for i,j in kb2id.items():
        if i == '南京南站':
            print(i, j)

    return id2kb,kb2id
def read_train(path=''):
    id2kb, kb2id=process_kb_data()
    chars={}
    all_alies = []
    with open('./data/ccks2019_el/train.json') as f:
        for l in tqdm(f):
            train_text=json.loads(l)
            text=train_text['text']
            for c in text:
                chars[c] = chars.get(c, 0) + 1
            s1 = [0] * len(text)
            s2 = [0] * len(text)
            temp=list()
            for x in train_text['mention_data']:
                if x['kb_id']!='NIL':
                    try:
                        kb_id=x['kb_id']
                        name=x['mention']
                        begin=int(x['offset'])
                        end=begin+len(name)-1
                        s1[begin]=1
                        s2[end]=1
                        y = [0] * len(text)
                        for i in range(begin, end+1):
                            y[i] = 1
                        name_ids=kb2id.get(name)
                        for i in name_ids:
                            if kb_id==i:
                                temp.append([text,name,y,'1',id2kb.get(i)['subject_desc']])
                            else:
                                temp.append([text, name,y,'0', id2kb.get(i)['subject_desc']])
                    except:
                        temp.append([text,name,y,'1',id2kb.get(kb_id)['subject_desc']])

            all_alies.extend([i + [s1] + [s2] for i in temp])
    return all_alies,id2kb,kb2id,chars
def save_file_to_file():
    all_alies,id2kb,kb2id,chars= read_train()
    for d in tqdm(iter(id2kb.values())):
        for c in d['subject_desc']:
            chars[c] = chars.get(c, 0) + 1
    chars = {i: j for i, j in chars.items() if j >= min_count}
    id2char = {i + 2: j for i, j in enumerate(chars)}
    char2id = {j: i for i, j in id2char.items()}
    with open('./data/kb.pkl', 'wb') as fw:
        pickle.dump(id2kb, fw)
        pickle.dump(kb2id, fw)
        pickle.dump(id2char, fw)
        pickle.dump(char2id, fw)
    with open('./data/train.pkl', 'wb') as fw:
        pickle.dump(all_alies, fw)
    print('finish!')

# save_file_to_file()
