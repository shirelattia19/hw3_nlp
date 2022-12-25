import os.path
import pickle

import numpy as np
import torch
from gensim.models import KeyedVectors
from torch import nn
from torch.nn import Sequential
from sklearn.neural_network import MLPRegressor
from torch.utils.data import Dataset, DataLoader
from statistics import mean

cposTable = ["PRP$", "VBG", "VBD", "VBN", ",", "''", "VBP", "WDT", "JJ", "WP", "VBZ", "DT", "#", "RP", "$", "NN", ")",
             "(", "FW", "POS", ".", "TO", "PRP", "RB", ":", "NNS", "NNP", "``", "WRB", "CC", "LS", "PDT", "RBS", "RBR",
             "CD", "EX", "IN", "WP$", "MD", "NNPS", "JJS", "JJR", "SYM", "VB", "UH", "ROOT-POS", "-LRB-", "-RRB-", '_']


class DependencyDataSet(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        with open(data_path, 'rb') as f:
            self.list_of_sentences = pickle.load(f)

    def __getitem__(self, item):
        return self.list_of_sentences[item]

    def __len__(self):
        return len(self.list_of_sentences)


# 'google_word2vec.model', 'trained_word2vec.model'
class DependencyEmbedding:
    def __init__(self, google_model_path, trained_model_path, i2w):
        self.google_word2vec = KeyedVectors.load(google_model_path)
        self.trained_word2vec = KeyedVectors.load(trained_model_path)
        self.embedding_dim = self.google_word2vec.vector_size + self.trained_word2vec.wv.vector_size + len(cposTable)
        self.i2w = i2w

    def __word_embedding(self, word_idx, pos_idx):
        if pos_idx != -1:
            word = self.i2w[word_idx]
            if word in self.google_word2vec.key_to_index:
                v_1 = self.google_word2vec[word]
            else:
                v_1 = self.google_word2vec['UNK']
            v_2 = self.trained_word2vec.wv[word]
            v_3 = np.zeros(len(cposTable))
            v_3[pos_idx] = 1
            return torch.from_numpy(np.concatenate((v_1, v_2, v_3)))
        else:
            return torch.zeros(self.embedding_dim)

    def sentence_embedding(self, word_tensor, pos_tensor):
        sentences = []
        for word_sen_tensor, pos_sen_tensor in zip(word_tensor, pos_tensor):
            sentence_embedding = []
            for word_idx, pos_idx in zip(word_sen_tensor, pos_sen_tensor):
                sentence_embedding.append(self.__word_embedding(word_idx.item(), pos_idx.item()))
            sentences.append(torch.stack(sentence_embedding))
        return torch.stack(sentences)


class DependencyParser(nn.Module):
    def __init__(self, hidden_dim, hidden_dim2, out_dim, alpha, i2w):
        super(DependencyParser, self).__init__()
        self.word_embedding = DependencyEmbedding('google_word2vec.model', 'trained_word2vec.model', i2w)
        self.input_dim = self.word_embedding.embedding_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.out_dim = out_dim
        self.tanh = nn.Tanh()
        self.encoder = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=2, bidirectional=True,
                               batch_first=True)  # TODO: dropout=self.dropout, batch_first=True ?
        self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim2, bias=True)
        self.fc2 = nn.Linear(self.hidden_dim2, self.out_dim, bias=True)
        self.dropout = nn.Dropout(alpha)
        self.edge_scorer = Sequential(self.dropout, self.fc1, self.tanh, self.fc2)
        # self.loss_function =  # Implement the loss function described above

    def forward(self, sentence):
        word_position_tensors, word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence.permute(1, 0, 2)

        # Pass word_idx through their embedding layer
        sentence_embedded = self.word_embedding.sentence_embedding(word_idx_tensor, pos_idx_tensor)

        # Get Bi-LSTM hidden representation for each word in sentence
        sentence_hidden_representation, _ = self.encoder(sentence_embedded.float())
        sentence_hidden_representation = self.tanh(sentence_hidden_representation)

        # Get score for each possible edge in the parsing graph, construct score matrix
        score_mat = self.edge_scorer(sentence_hidden_representation)

        if true_tree_heads[0][0].item() == -1:
            return None, score_mat
        else:
            # # Calculate the negative log likelihood loss described above
            # loss = self.loss_function(score_mat, true_tree_heads)

            # return loss, score_mat
            return None, score_mat


def train(model, data_sets, optimizer, num_epochs: int, hp, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
                    "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False)}
    model.to(device)
    best_uas = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        loss_history_train_epoch = []
        loss_history_valid_epoch = []
        uas_train_epoch = []
        uas_valid_epoch = []
        # acc_valid_epoch = []

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for batch in data_loaders[phase]:
                if phase == 'train':
                    optimizer.zero_grad()
                    loss, score_mat = model(batch.to(device))
                    loss.backward()
                    optimizer.step()
                    loss_history_train_epoch.append(loss)

                    uas = 0  # TODO: calculate uas
                    uas_train_epoch.append(uas)
                else:
                    with torch.no_grad():
                        loss, score_mat = model(batch.to(device))
                        loss_history_valid_epoch.append(loss)
                        uas = 0  # TODO: calculate
                        uas_valid_epoch.append(uas)
                        # acc_valid_epoch.append((batch[1] == pred).float().sum()/(pred.shape[0]*pred.shape[1]))

            if phase == 'train':
                epoch_loss_train = torch.mean(torch.stack(loss_history_train_epoch))
                epoch_uas_Score_train = mean(uas_train_epoch)
                print(f'{phase.title()} Train Loss: {epoch_loss_train:.4e} Train uas score: {epoch_uas_Score_train}')
            else:
                epoch_loss_valid = torch.mean(torch.stack(loss_history_valid_epoch))
                epoch_uas_Score_valid = mean(uas_valid_epoch)
                # epoch_acc_valid = torch.mean(torch.stack(acc_valid_epoch))
                print(f'{phase.title()} Valid Loss: {epoch_loss_valid:.4e} Valid uas score: {epoch_uas_Score_valid} ')
                # f'Valid acc: {epoch_acc_valid}')

                if epoch_uas_Score_valid > best_uas:
                    best_uas = epoch_uas_Score_valid
                    with open(os.path.join('checkpoints', str(hp), f'model_{best_uas}.pkl'), 'wb') as f:
                        torch.save(model, f)

    print(f'Best Validation uas score: {best_uas:4f}')
    return best_uas


def predict(model, comp_dataset, batch_size, test_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = DataLoader(comp_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    pred = []
    for batch in data_loader:
        _, score_mat = model(batch.to(device))
        # pred_batch = ????
        # pred.append(pred_batch)
    pred = torch.cat(pred)

    # Create the tagged file from untagged
    # writes_tagged_test(pred, test_path, output_path)


from chu_liu_edmonds import decode_mst


def eval_model(model, sentence):
    _, score_mat = model(sentence)
    predicted_tree = decode_mst(score_mat)
    return predicted_tree
