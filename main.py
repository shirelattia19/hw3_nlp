import os.path
import re
import numpy as np
from gensim import downloader
from gensim.models import Word2Vec
import pickle
import torch

from torch.optim import Adam

from LSTM import DependencyParser, DependencyDataSet, train, cposTable, DependencyEmbedding


def preprocess(path, w2i, word_embedding):
    list_of_sentences_with_tags = []
    sentence = []
    tags = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line == "\n":
                list_of_sentences_with_tags.append([torch.stack(sentence), torch.Tensor(tags)])
                sentence = []
                tags = []
                continue
            if line[-1:] == "\n":
                line = line[:-1]
            line_set = re.split(r'\t+', line)
            tag = line_set[6] if line_set[6] != '_' else -1
            word = line_set[1]
            word_pos = line_set[3]
            word_idx = w2i[word]
            pos_idx = cposTable.index(word_pos)
            word_embedded = word_embedding.word_embedding(word_idx, pos_idx)
            sentence.append(word_embedded)
            tags.append(int(tag))
    return list_of_sentences_with_tags


def preprocess_first(path):
    sentence_index = 0
    list_of_sentences_with_tags = []
    sentence = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line == "\n":
                sentence_index += 1
                list_of_sentences_with_tags.append(sentence)
                sentence = []
                continue
            if line[-1:] == "\n":
                line = line[:-1]
            line_set = re.split(r'\t+', line)
            tag = line_set[6] if line_set[6] != '_' else -1
            word = line_set[1]
            word_pos = line_set[3]
            X_representation = np.array([word, cposTable.index(word_pos), int(tag)])
            sentence.append(X_representation)
    list_of_sentences_with_tags = [np.array(sen).T for sen in list_of_sentences_with_tags]
    return list_of_sentences_with_tags


def create_data(train_path, test_path, com_path, w2i, i2w):
    word_embedding = DependencyEmbedding('google_word2vec.model', 'trained_word2vec.model', i2w)

    list_of_sentences_with_tags_train = preprocess(train_path, w2i, word_embedding)
    with open(f"train.emb", 'wb+') as f:
        pickle.dump(list_of_sentences_with_tags_train, f)

    list_of_sentences_with_tags_test = preprocess(test_path, w2i, word_embedding)
    with open(f"test.emb", 'wb+') as f:
        pickle.dump(list_of_sentences_with_tags_test, f)

    list_of_sentences_comp = preprocess(com_path, w2i, word_embedding)
    with open(f"comp.emb", 'wb+') as f:
        pickle.dump(list_of_sentences_comp, f)


def create_data_first(train_path, test_path, com_path):
    list_of_sentences_with_tags_train = preprocess_first(train_path)
    with open(f"train.data", 'wb+') as f:
        pickle.dump(list_of_sentences_with_tags_train, f)

    list_of_sentences_with_tags_test = preprocess_first(test_path)
    with open(f"test.data", 'wb+') as f:
        pickle.dump(list_of_sentences_with_tags_test, f)

    list_of_sentences_comp = preprocess_first(com_path)
    with open(f"comp.data", 'wb+') as f:
        pickle.dump(list_of_sentences_comp, f)


def create_w2i_i2w(path_list):
    words = []
    for path in path_list:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line == "\n":
                    continue
                if line[-1:] == "\n":
                    line = line[:-1]
                line_set = re.split(r'\t+', line)
                word = line_set[1]
                words.append(word)

    words = list(set(words))
    w2i = {k: v for v, k in enumerate(words)}
    i2w = {v: k for v, k in enumerate(words)}
    with open(f"w2i.dict", 'wb+') as f:
        pickle.dump(w2i, f)
    with open(f"i2w.dict", 'wb+') as f:
        pickle.dump(i2w, f)


def pre_embedding():
    # WORD_2_VEC_PATH = 'word2vec-google-news-300'
    # google_model = downloader.load(WORD_2_VEC_PATH)
    # google_model.save("google_word2vec.model")

    with open(f"train.data", 'rb') as f:
        list_of_sentences_with_tags_train = pickle.load(f)
    with open(f"test.data", 'rb') as f:
        list_of_sentences_with_tags_test = pickle.load(f)
    with open(f"comp.data", 'rb') as f:
        list_of_sentences_with_tags_comp = pickle.load(f)
    sentences = [list(sen[0]) for sen in (
            list_of_sentences_with_tags_train + list_of_sentences_with_tags_test + list_of_sentences_with_tags_comp)]
    trained_model = Word2Vec(sentences=sentences, vector_size=100, window=2, min_count=1, workers=4, epochs=100,
                             seed=42)
    trained_model.save("trained_word2vec.model")


if __name__ == '__main__':
    train_path = f'train.labeled'
    test_path = f'test.labeled'
    com_path = f'comp.unlabeled'

    # # 1- Create w2i and i2w dicts
    # create_w2i_i2w([train_path, test_path, com_path])

    with open(f"w2i.dict", 'rb') as f:
        w2i = pickle.load(f)
    with open(f"i2w.dict", 'rb') as f:
        i2w = pickle.load(f)

    # # 2- Preprocess the data files
    # create_data_first(train_path, test_path, com_path)

    # # 3- Create the embedding models
    # pre_embedding()

    # # 4- create the embedded data files for efficiency
    # create_data(train_path, test_path, com_path, w2i, i2w)


    # 5- train model
    hp = dict(num_epochs=100, hidden_dim=256, hidden_dim2=128, alpha=0.25, lr=0.004, grad_step_num=130,
              percentage_of_data=1)

    train_ds = DependencyDataSet(f"train.emb", hp['percentage_of_data'])
    test_ds = DependencyDataSet(f"test.emb", hp['percentage_of_data'])
    datasets = {"train": train_ds, "test": test_ds}
    #
    model = DependencyParser(hidden_dim=hp['hidden_dim'], hidden_dim2=hp['hidden_dim2'], alpha=hp['alpha'], i2w=i2w)
    optimizer = Adam(params=model.parameters(), lr=hp['lr'])
    best_uas = train(model=model, data_sets=datasets, optimizer=optimizer, num_epochs=hp['num_epochs'],
                     grad_step_num=hp["grad_step_num"], hp=hp)
