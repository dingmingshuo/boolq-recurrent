from utils import yaml_load
from data import get_train_data, get_dev_data, pad_sequence
from model import RecurrentModel

import os

import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torch.optim import Adam
from tqdm import tqdm

from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_auc_score

def collate_fn(data):
    passages_ids = []
    questions_ids = []

    passages_length = []
    questions_length = []

    passages_mask = []
    questions_mask = []

    answers = []

    max_passage_length = 0
    max_question_length = 0
    for(passage_ids, question_ids, answer) in data:
        max_passage_length = max(max_passage_length, len(passage_ids))
        max_question_length = max(max_question_length, len(question_ids))

    max_passage_length = min(
        max_passage_length,  preprocess_cfg['pa_max_sent_len'])
    max_question_length = min(
        max_question_length,  preprocess_cfg['qu_max_sent_len'])

    for(passage_ids, question_ids, answer) in data:
        passages_ids.append(pad_sequence(
            passage_ids, max_passage_length))
        questions_ids.append(pad_sequence(
            question_ids, max_question_length))

        passages_length.append(len(passage_ids))
        questions_length.append(len(question_ids))

        passages_mask.append(
            np.concatenate((np.ones(len(passage_ids)), np.zeros(max_passage_length-len(passage_ids)))))
        questions_mask.append(
            np.concatenate((np.ones(len(question_ids)), np.zeros(max_question_length-len(question_ids)))))

        answers.append(answer)

    return passages_ids, questions_ids, passages_length, questions_length, passages_mask, questions_mask, answers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = yaml_load("config.yaml")
model_cfg = config.get("model", {})
preprocess_cfg = config.get("preprocess", {})
data_cfg = config.get("data", {})
dev_cfg = config.get("dev", {})
train_cfg = config.get("train", {})
eval_cfg = config.get("eval", {})

vocab = {}
with open(data_cfg['vocab_file'], 'rb') as handle:
    vocab = pickle.load(handle)
handle.close()

dev_data = get_dev_data(
    data_cfg['data_path'], data_cfg['dev_data'], vocab, [preprocess_cfg['pa_max_sent_len'], preprocess_cfg['qu_max_sent_len']])

model = RecurrentModel(
    preprocess_cfg['vocab_size'], model_cfg['embedding_dim'], model_cfg['hidden_size']).to(device)
checkpoint = torch.load(eval_cfg['ckpt_path'], map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print(model)

eval_loader = DataLoader(
    dev_data,
    batch_size=dev_cfg["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

# Evaluate
pred = np.array([])
answer = np.array([])
prob = np.array([])

with torch.no_grad():
    for passages_ids, questions_ids, passages_length, questions_length, passages_mask, questions_mask, labels in tqdm(eval_loader):
        passages_ids = torch.tensor(passages_ids).long().to(device)
        questions_ids = torch.tensor(questions_ids).long().to(device)
        passages_length = torch.tensor(passages_length).long()
        questions_length = torch.tensor(questions_length).long()
        passages_mask = torch.tensor(passages_mask).long().to(device)
        questions_mask = torch.tensor(questions_mask).long().to(device)
        labels = torch.tensor(labels).to(device)

        eval_outputs = model(passages_ids, questions_ids, passages_length,
                        questions_length, passages_mask, questions_mask)        
        results = softmax(eval_outputs, dim=1)
        now_pred = torch.argmax(results, dim=1)
        pred = np.concatenate((pred, now_pred.cpu().numpy()), axis=0)
        answer = np.concatenate((answer, labels.cpu().numpy()), axis=0)
        prob = np.concatenate((prob, results.cpu().numpy()[:,1]), axis=0)

# Calculate metric scores
accuracy = accuracy_score(answer, pred)
recall = recall_score(answer, pred)
precision = precision_score(answer, pred)
f1 = f1_score(answer, pred)
roc_auc = roc_auc_score(answer, prob)

print("accuracy: ", accuracy)
print("recall: ", recall)
print("precision: ", precision)
print("f1 score: ", f1)
print("roc_auc score: ", roc_auc)

# Save results
np.savetxt(eval_cfg['result_path'], pred, fmt = "%d", delimiter = "\n")