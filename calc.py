import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
from collections import Counter

from utils import yaml_load

config = yaml_load("./config.yaml")

data_cfg = config.get("data",{})

words = []
passage_len = []
question_len = []

data_file = os.path.join(data_cfg['data_path'],data_cfg['train_data'])
print(data_file)
data_df = pd.read_json(data_file,lines = True,orient='records')

passages = data_df.passage.values
questions = data_df.question.values

for passage,question in tqdm(zip(passages,questions)):
    passage_words = word_tokenize(passage.lower())
    question_words = word_tokenize(question.lower())
    
    words.extend(passage_words)
    words.extend(question_words)
    
    passage_len.append(len(passage_words))
    question_len.append(len(question_words))

words_cnt = Counter(words)
passage_len = np.array(passage_len)
question_len = np.array(question_len)

with open(data_cfg['calc_file'],'w',encoding='utf-8') as f:
    for (word,freq) in words_cnt.most_common():
        f.write(word+":{}\n".format(freq))
f.close()

print(np.max(passage_len),np.mean(passage_len),np.std(passage_len))
print(np.max(question_len),np.mean(question_len),np.std(question_len))

word_idx = {'<UNK>':1,'<STA>':2,'<END>':3}
cnt = 4
for (word,freq) in words_cnt.most_common():
    word_idx[word] = cnt
    cnt += 1
    if cnt == 2500:
        break

with open(data_cfg['vocab_file'],'wb') as handle:
    pickle.dump(word_idx,handle)
handle.close()




