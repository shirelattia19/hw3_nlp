import pickle

import numpy as np
import torch
from gensim.models import KeyedVectors
from torch import nn
from torch.nn import Sequential
from torch.utils.data import Dataset
from sklearn.neural_network import MLPRegressor

cposTable = ["PRP$", "VBG", "VBD", "VBN", ",", "''", "VBP", "WDT", "JJ", "WP", "VBZ", "DT", "#", "RP", "$", "NN", ")",
             "(", "FW", "POS", ".", "TO", "PRP", "RB", ":", "NNS", "NNP", "``", "WRB", "CC", "LS", "PDT", "RBS", "RBR",
             "CD", "EX", "IN", "WP$", "MD", "NNPS", "JJS", "JJR", "SYM", "VB", "UH", "ROOT-POS", "-LRB-", "-RRB-"]


class DependencyDataSet(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        with open(f"train.data", 'rb') as f:
            self.list_of_sentences = pickle.load(f)

    def __getitem__(self, item):
        return self.list_of_sentences[item]

    def __len__(self):
        return len(self.list_of_sentences)


# 'google_word2vec.model', 'trained_word2vec.model'
class DependencyEmbedding:
    def __init__(self, google_model_path, trained_model_path):
        self.google_word2vec = KeyedVectors.load(google_model_path)
        self.trained_word2vec = KeyedVectors.load(trained_model_path)
        self.embedding_dim = self.google_word2vec.vector_size + self.trained_word2vec.wv.vector_size + len(cposTable)

    def __word_embedding(self, word, pos):
        if word in self.google_word2vec.key_to_index:
            v_1 = self.google_word2vec[word]
        else:
            v_1 = self.google_word2vec['UNK']
        v_2 = self.trained_word2vec.wv[word]
        pos_idx = cposTable.index(pos)
        v_3 = np.zeros(len(cposTable))
        v_3[pos_idx] = 1
        return np.concatenate((v_1, v_2, v_3))

    def sentence_embedding(self, word_tensor, pos_tensor):
        sentence_embedding = []
        for idx, (word, pos) in enumerate(zip(word_tensor, pos_tensor)):
            sentence_embedding.append(self.__word_embedding(word, pos))
        return sentence_embedding


class DependencyParser(nn.Module):
    def __init__(self, hidden_dim, hidden_dim2, out_dim, alpha):
        super(DependencyParser, self).__init__()
        self.word_embedding = DependencyEmbedding('google_word2vec.model', 'trained_word2vec.model', )
        self.input_dim = self.word_embedding.embedding_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.out_dim = out_dim
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=2, bidirectional=True,
                               batch_first=False)
        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim2, bias=True)
        self.fc2 = nn.Linear(self.hidden_dim2, self.out_dim, bias=True)
        self.dropout = nn.Dropout(alpha)
        self.edge_scorer = Sequential(self.dropout, self.fc1, self.tanh, self.fc2)
    def NLLL(self,score_mat,label):
        out = torch.diag(score_mat[:, label])
        return -torch.mean(out)


    def forward(self, sentence,labels):
        word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence

        # Pass word_idx through their embedding layer
        sentence_embedded = self.word_embedding.sentence_embedding(word_idx_tensor, pos_idx_tensor)

        # Get Bi-LSTM hidden representation for each word in sentence
        sentence_hidden_representation = []
        for word_embedded in sentence_embedded:
            word_hidden_representation = self.encoder(word_embedded)
            word_hidden_representation = self.tanh(word_hidden_representation)
            sentence_hidden_representation.append(word_hidden_representation)

        # Get score for each possible edge in the parsing graph, construct score matrix
        score_mat = self.edge_scorer(sentence_hidden_representation)
        print(score_mat.shape)
        # # Calculate the negative log likelihood loss described above
        loss = self.NLLL(score_mat, labels)

        # return loss, score_mat
        return loss, score_mat


from chu_liu_edmonds import decode_mst


def eval_model(model, sentence):
    _, score_mat = model(sentence)
    predicted_tree = decode_mst(score_mat)
    return predicted_tree
