import pickle
from utils import yaml_load


def get_numpy_word_embed():
    row = 0

    config = yaml_load("config.yaml")
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    glove_path = model_cfg["glove_path"]
    glove_length = model_cfg["glove_length"]
    vocab_file = data_cfg["vocab_file"]

    words_embed = {}
    with open(glove_path, mode='r')as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            embed = [float(num) for num in embed]
            words_embed[word] = embed
            if row > 20000:
                break
            row += 1

    word2idx = {}
    with open(vocab_file, 'rb') as handle:
        word2idx = pickle.load(handle)
    idx2word = {idx: w for w, idx in word2idx.items()}
    id2emb = {}
    id2emb[0] = [0.0] * glove_length
    for (_, idx) in word2idx.items():
        if idx2word[idx] in words_embed:
            id2emb[idx] = words_embed[idx2word[idx]]
        else:
            id2emb[idx] = [0.0] * glove_length
    data = [id2emb[idx] for idx in range(len(word2idx) + 1)]

    return data
