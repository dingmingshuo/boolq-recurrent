from utils import yaml_load
from data import get_train_data, get_dev_data, pad_sequence
from model import RecurrentModel

import os

import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = yaml_load("config.yaml")
model_cfg = config.get("model", {})
preprocess_cfg = config.get("preprocess", {})
data_cfg = config.get("data", {})
dev_cfg = config.get("dev", {})
train_cfg = config.get("train", {})


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


vocab = {}
with open(data_cfg['vocab_file'], 'rb') as handle:
    vocab = pickle.load(handle)
handle.close()

train_data = get_train_data(
    data_cfg['data_path'], data_cfg['train_data'], vocab, [preprocess_cfg['pa_max_sent_len'], preprocess_cfg['qu_max_sent_len']])
dev_data = get_dev_data(
    data_cfg['data_path'], data_cfg['dev_data'], vocab, [preprocess_cfg['pa_max_sent_len'], preprocess_cfg['qu_max_sent_len']])

model = RecurrentModel(
    preprocess_cfg['vocab_size'], model_cfg['embedding_dim'], model_cfg['hidden_size']).to(device)
model.train()
print(model)

optimizer = Adam(model.parameters(), train_cfg["lr"],
                 weight_decay=train_cfg["weight_decay"])
crossentropyloss = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(
    train_data,
    batch_size=train_cfg["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

dev_loader = DataLoader(
    dev_data,
    batch_size=dev_cfg["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

logging_step = train_cfg["logging_step"]
b = train_cfg["b"]

for epoch in range(train_cfg["epochs"]):
    total_loss = 0
    step_now = 0
    dev_loss = 0
    with tqdm(train_loader) as tl:
        for passages_ids, questions_ids, passages_length, questions_length, passages_mask, questions_mask, answers in tl:
            step_now += 1
            optimizer.zero_grad()

            passages_ids = torch.tensor(passages_ids).long().to(device)
            questions_ids = torch.tensor(questions_ids).long().to(device)
            passages_length = torch.tensor(passages_length).long()
            questions_length = torch.tensor(questions_length).long()
            passages_mask = torch.tensor(passages_mask).long().to(device)
            questions_mask = torch.tensor(questions_mask).long().to(device)
            answers = torch.tensor(answers).to(device)

            outputs = model(passages_ids, questions_ids, passages_length,
                            questions_length, passages_mask, questions_mask)

            loss = (crossentropyloss(outputs, answers) - b).abs() + b
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()

            # Evaluate
            if step_now % logging_step == 0:
                dev_loss = 0
                model.eval()
                with torch.no_grad():
                    for passages_ids, questions_ids, passages_length, questions_length, passages_mask, questions_mask, answers in dev_loader:
                        passages_ids = torch.tensor(
                            passages_ids).long().to(device)
                        questions_ids = torch.tensor(
                            questions_ids).long().to(device)
                        passages_length = torch.tensor(passages_length).long()
                        questions_length = torch.tensor(
                            questions_length).long()
                        passages_mask = torch.tensor(
                            passages_mask).long().to(device)
                        questions_mask = torch.tensor(
                            questions_mask).long().to(device)
                        answers = torch.tensor(answers).to(device)

                        outputs = model(passages_ids, questions_ids, passages_length,
                                        questions_length, passages_mask, questions_mask)
                        loss = crossentropyloss(outputs, answers)

                        dev_loss += loss.cpu().item()
                model.train()

            # Load loggings
            tl.set_postfix(loss=loss.cpu().item(),
                           avg_loss=total_loss/step_now,
                           dev_loss=dev_loss/len(dev_loader))

    # Save model
    if not os.path.isdir(train_cfg["output_dir"]):
        os.mkdir(train_cfg["output_dir"])
    output_path = os.path.join(
        train_cfg["output_dir"], train_cfg["output_filename_prefix"])
    torch.save(model.state_dict(), (output_path+"_epoch=%d") % (epoch+1))
