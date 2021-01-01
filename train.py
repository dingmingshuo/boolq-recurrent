from utils import yaml_load
from data import get_train_data, get_dev_data,pad_sequence
from model import RecurrentModel

import os

import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

def collate_fn(data):
    passages_ids = []
    questions_ids = []

    passages_length = []
    questions_length = []

    passages_mask = []
    questions_mask = []    
    
    answers = []
    
    for(passage_ids,question_ids,answer) in data:
        passages_ids.append(pad_sequence(passage_ids,preprocess_cfg['pa_max_sent_len']))
        questions_ids.append(pad_sequence(question_ids,preprocess_cfg['qu_max_sent_len']))

        passages_length.append(len(passage_ids))
        questions_length.append(len(question_ids))
        
        passages_mask.append(
            np.concatenate(np.ones(len(passage_ids)),np.zeros(preprocess_cfg['pa_max_sent_len']-len(passage_ids))))
        questions_mask.append(
            np.concatenate(np.ones(len(question_ids)),np.zeros(preprocess_cfg['qu_max_sent_len']-len(question_ids))))
        
        answers.append(answer)
    
    return passages_ids,questions_ids,passages_length,questions_length,passages_mask,questions_mask,answers

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

config = yaml_load("./config.yaml")
model_cfg = config.get("model")
preprocess_cfg = config.get("preprocess", {})
data_cfg = config.get("data", {})
dev_cfg = config.get("dev", {})
train_cfg = config.get("train", {})

vocab = {}
with open(data_cfg['vocab_file'],'rb') as handle:
    vocab = pickle.load(handle)
handle.close()

train_data = get_train_data(
    data_cfg['data_path'], data_cfg['train_data'], vocab, [preprocess_cfg['pa_max_sent_len'],preprocess_cfg['qu_max_sent_len']])
dev_data = get_dev_data(
    data_cfg['data_path'], data_cfg['dev_data'], vocab, [preprocess_cfg['pa_max_sent_len'],preprocess_cfg['qu_max_sent_len']])

model = RecurrentModel(preprocess_cfg['vocab_size'],model_cfg['embedding_dim'],model_cfg['hidden_size']).to(device)
model.train()

optimizer = Adam(model, train_cfg["lr"], weight_decay = train_cfg["weight_decay"])
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
        for passages_ids,questions_ids,passages_length,questions_length,passages_mask,questions_mask,answers in tl:
            step_now += 1
            optimizer.zero_grad()
            
            passages_ids = torch.Tensor(passages_ids).to(device)
            questions_ids = torch.Tensor(questions_ids).to(device)
            passages_length = torch.Tensor(passages_length).to(device)
            questions_length = torch.Tensor(questions_length).to(device)
            passages_mask = torch.Tensor(passages_mask).long().to(device)
            questions_mask = torch.Tensor(questions_mask).long().to(device)
            answers = torch.Tensor(answers).to(device)
            
            outputs = model(passages_ids,questions_ids,passages_length,questions_length,passages_mask,questions_mask)


            loss = (crossentropyloss(outputs,answers) - b).abs() + b
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()

            # Evaluate
            if step_now % logging_step == 0:
                dev_loss = 0
                model.eval()
                with torch.no_grad():
                    for passages_ids,questions_ids,passages_length,questions_length,passages_mask,questions_mask,answers in tl:    
                        passages_ids = torch.Tensor(passages_ids).to(device)
                        questions_ids = torch.Tensor(questions_ids).to(device)
                        passages_length = torch.Tensor(passages_length).to(device)
                        questions_length = torch.Tensor(questions_length).to(device)
                        passages_mask = torch.Tensor(passages_mask).long().to(device)
                        questions_mask = torch.Tensor(questions_mask).long().to(device)
                        answers = torch.Tensor(answers).to(device)
                        
                        outputs = model(passages_ids,questions_ids,passages_length,questions_length,passages_mask,questions_mask)
                        loss = crossentropyloss(outputs,answers)
                        
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
