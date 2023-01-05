import os
import pickle

import numpy as np
import torch
from gensim.models import KeyedVectors
from torch import nn
from torch.nn import Sequential
from torch.utils.data import Dataset, DataLoader
from statistics import mean

cposTable = ["PRP$", "VBG", "VBD", "VBN", ",", "''", "VBP", "WDT", "JJ", "WP", "VBZ", "DT", "#", "RP", "$", "NN", ")",
             "(", "FW", "POS", ".", "TO", "PRP", "RB", ":", "NNS", "NNP", "``", "WRB", "CC", "LS", "PDT", "RBS", "RBR",
             "CD", "EX", "IN", "WP$", "MD", "NNPS", "JJS", "JJR", "SYM", "VB", "UH", "ROOT-POS", "-LRB-", "-RRB-", '_']


class DependencyDataSet(Dataset):
    def __init__(self, data_path, percentage_of_data=None):
        self.data_path = data_path
        with open(data_path, 'rb') as f:
            self.list_of_sentences = pickle.load(f)
        if percentage_of_data is not None:
            index = int(len(self.list_of_sentences) * percentage_of_data)
            self.list_of_sentences = self.list_of_sentences[0:index]

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

    def word_embedding(self, word_idx, pos_idx):
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
        sentence_embedding = []
        for word_idx, pos_idx in zip(word_tensor, pos_tensor):
            sentence_embedding.append(self.word_embedding(word_idx.item(), pos_idx.item()))
        return torch.stack(sentence_embedding)


class DependencyParser(nn.Module):
    def __init__(self, hidden_dim, hidden_dim2, alpha, i2w):
        super(DependencyParser, self).__init__()
        self.i2w = i2w
        self.word_embedding = DependencyEmbedding('google_word2vec.model', 'trained_word2vec.model', i2w)
        self.input_dim = self.word_embedding.embedding_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(alpha)
        self.encoder = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=2, bidirectional=True,
                               batch_first=True, dropout=alpha)
        self.encoder_2 = nn.LSTM(self.hidden_dim * 2, self.hidden_dim, num_layers=2, bidirectional=True,
                                 dropout=alpha, batch_first=True)

        # self.fc1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim2, bias=True)
        # self.edge_scorer = None
        # self.fc2 = None
        # Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph
        self.mlp_heads = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim2)
        )
        self.mlp_modifiers = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim2)
        )

        self.edge_scorer = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.hidden_dim2, 1)
        )
        # self.loss_function =  # Implement the loss function described above

    def NLLLoss(self, score_mat, true_heads, sentence_len, device):
        # true heads without root
        # score mat  avec root row et column
        # true_heads = true_heads.long()
        # res = -torch.sum(score_mat[range(true_heads.shape[0]), true_heads.long()]) / true_heads.shape[0]
        # return res
        # -torch.sum(torch.log(score_mat)[range(true_heads.shape[0]), true_heads]) / true_heads.shape[0]
        true_heads = true_heads.long()
        predicted_scores = score_mat[:, 1:]
        loss = torch.zeros(1, device=device)
        cross_entropy_loss = nn.CrossEntropyLoss()
        for modifier_idx in range(sentence_len):
            edge = predicted_scores[:, modifier_idx].unsqueeze(dim=0)
            head_idx = modifier_idx + 1
            true_score = true_heads[modifier_idx:head_idx]
            cross = cross_entropy_loss(edge, true_score)
            loss += cross
        return (1.0 / sentence_len) * loss

    def run_edge_scorer(self, lstm_output, sentence_len):
        lstm_output = lstm_output.squeeze()

        # Get score for each possible edge in the parsing graph, construct score matrix
        score_matrix = torch.FloatTensor(sentence_len, sentence_len)

        # run over all of the heads with the MLP_head
        words_as_heads = self.mlp_heads(lstm_output)
        # run over all of the heads with the MLP_modifier
        words_as_modifiers = self.mlp_modifiers(lstm_output)

        # loop over heads
        for head_idx in range(sentence_len):
            # loop over modifiers
            for modifier_idx in range(sentence_len):
                if head_idx == modifier_idx:
                    score_matrix[head_idx][modifier_idx] = 0
                    continue
                score_matrix[head_idx][modifier_idx] = self.edge_scorer(
                    words_as_heads[head_idx] + words_as_modifiers[modifier_idx])
        return score_matrix

    def forward(self, sentence, device):
        # return value: loss: tensor of 1 with grad, mst: seq_len + (with the -1 of the root)

        # Initialization
        word_position_tensors, word_idx_tensor, pos_idx_tensor, true_tree_heads = torch.squeeze(sentence)
        if len(word_position_tensors.shape) == 0:
            word_position_tensors = torch.unsqueeze(word_position_tensors, dim=0)
            word_idx_tensor = torch.unsqueeze(word_idx_tensor, dim=0)
            pos_idx_tensor = torch.unsqueeze(pos_idx_tensor, dim=0)
            true_tree_heads = torch.unsqueeze(true_tree_heads, dim=0)
        out_dim = len(word_position_tensors) + 1
        # self.fc2 = nn.Linear(self.hidden_dim2, out_dim, bias=True)
        # self.edge_scorer = Sequential(self.dropout, self.fc1, self.tanh, self.fc2)

        # Pass word_idx through their embedding layer
        sentence_embedded = self.word_embedding.sentence_embedding(word_idx_tensor, pos_idx_tensor)

        # Get Bi-LSTM hidden representation for each word in sentence
        sentence_hidden_representation, _ = self.encoder(sentence_embedded.float().to(device))
        #sentence_hidden_representation = self.tanh(sentence_hidden_representation)
        sentence_hidden_representation, _ = self.encoder_2(sentence_hidden_representation.to(device))
        sentence_hidden_representation = torch.cat(
            (torch.zeros((1, sentence_hidden_representation.shape[1]), device=device), sentence_hidden_representation))

        # Get score for each possible edge in the parsing graph, construct score matrix
        # score_mat = self.edge_scorer(sentence_hidden_representation)
        score_mat = self.run_edge_scorer(sentence_hidden_representation, out_dim)

        mst, _ = decode_mst(score_mat.cpu().detach(), out_dim, False)
        mst = torch.from_numpy(mst)
        if true_tree_heads[0].item() == -1:
            return None, mst[1:]
        else:
            # Calculate the negative log likelihood loss described above
            loss = self.NLLLoss(score_mat.to(device), true_tree_heads.to(device), out_dim - 1, device)
            return loss, mst[1:]


def train(model, data_sets, optimizer, num_epochs: int, grad_step_num, hp):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=1, shuffle=True),
                    "test": DataLoader(data_sets["test"], batch_size=1, shuffle=False)}
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

            for batch_idx, batch in enumerate(data_loaders[phase]):
                if phase == 'train':
                    if batch_idx % grad_step_num == 0 and batch_idx != 0:
                        optimizer.zero_grad()

                    loss, mst = model(batch.to(device), device)
                    loss = loss / grad_step_num
                    loss.backward()

                    if batch_idx % grad_step_num == 0 and batch_idx != 0:
                        optimizer.step()

                    loss_history_train_epoch.append(loss)
                    true_tree = torch.squeeze(batch)[3]
                    if len(true_tree.shape) == 0:
                        true_tree = torch.unsqueeze(true_tree, dim=0)
                    try:
                        uas = uas_compute(mst, true_tree)
                    except:
                        print('tt')
                    uas_train_epoch.append(uas)
                else:
                    with torch.no_grad():
                        loss, mst = model(batch.to(device), device)
                        loss_history_valid_epoch.append(loss)

                        true_tree = torch.squeeze(batch)[3]
                        uas = uas_compute(mst, true_tree)
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
                    # checkpoint_path = os.path.join('checkpoints', '_'.join([f"{key}_{value}" for (key, value) in hp.items()]))
                    # os.makedirs(checkpoint_path)
                    # with open(os.path.join(checkpoint_path, f'model_{best_uas}.pkl'), 'wb') as f:
                    #     torch.save(model, f)

    print(f'Best Validation uas score: {best_uas:4f}')
    return best_uas


def uas_compute(mst, true_tree):
    res = torch.eq(mst, true_tree)
    res = torch.sum(res)
    if len(mst) != 1:
        res = (res.item() / (len(mst) - 1))
    else:
        res = res.item()
    return res * 100


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
