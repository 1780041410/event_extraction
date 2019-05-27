import pandas as pd
import pickle
df_test = pd.read_csv('./data/event_type_entity_extract_eval.csv',header=None)
print(df_test.head())
#
da = pd.read_table('./result.txt',header = None,delim_whitespace=True)



df_3=pd.DataFrame()

df_3=pd.concat([df_test[0],da[1]],axis=1)
print(df_3)
df_3.to_csv('result2.txt',header = None,sep=',',index=False,na_rep="NaN")

# with open('./result.txt','r') as fr:
#     with open('result_2.txt','w',encoding='utf-8') as fw:
#         for line in fr:
#             if len(line.strip().split())==2:
#                 fw.write(line)
#             else:
#                 fw.write(str(line.strip().split()[0])+'\t'+'NaN'+'\n')