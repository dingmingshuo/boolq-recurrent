import os

import pickle
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class BoolQDataset(Dataset):
    def __init__(self, passage_ids,question_ids,answers):
        self.passage_ids = passage_ids
        self.question_ids = question_ids
        self.answers = answers


    def __getitem__(self, index):
        passage_ids = torch.Tensor(self.passage_ids[index],dtype=torch.long)
        question_ids = torch.Tensor(self.question_ids[index],dtype=torch.long)
        answer = torch.tensor(self.answers[index], dtype=torch.long)
        return passage_ids,question_ids,answer

    def __len__(self):
        return len(self.answers)

def encode_data(vocab, texts, max_length):
    
    text_ids = []

    for text in tqdm(texts):
        words = nltk.word_tokenize(text.lower())
        words.insert(0,'<STA>')
        words.append('<END>')
        ids = [vocab[word] if word in vocab else 1 for word in words[:max_length]]

        text_ids.append(ids)

    return np.array(text_ids)

def get_train_data(data_path, train_data_file, vocab, max_seq_length):
    train_data_file = os.path.join(data_path, train_data_file)
    train_data_df = pd.read_json(train_data_file, lines=True, orient='records')

    passages_train = train_data_df.passage.values
    questions_train = train_data_df.question.values
    answers_train = train_data_df.answer.values.astype(int)

    print("Importing train datas from %s:" % train_data_file)
    passages_ids_train = encode_data(
        vocab, passages_train, max_seq_length[0])
    questions_ids_train = encode_data(
        vocab, questions_train, max_seq_length[1])

    return BoolQDataset(passages_ids_train, questions_ids_train, answers_train)

def get_dev_data(data_path, dev_data_file, vocab, max_seq_length):
    dev_data_file = os.path.join(data_path, dev_data_file)
    dev_data_df = pd.read_json(dev_data_file, lines=True, orient='records')

    passages_dev = dev_data_df.passage.values
    questions_dev = dev_data_df.question.values
    answers_dev = dev_data_df.answer.values.astype(int)

    print("Importing dev datas from %s:" % dev_data_file)
    passages_ids_dev = encode_data(
        vocab, passages_dev, max_seq_length[0])
    questions_ids_dev = encode_data(
        vocab, questions_dev, max_seq_length[1])

    return BoolQDataset(passages_ids_dev, questions_ids_dev, answers_dev)

def pad_sequence(ids,max_length):
    if len(ids) <max_length:
        return np.concatenate((ids,np.zeros(max_length-len(ids)).astype(np.int64)))
    return ids
