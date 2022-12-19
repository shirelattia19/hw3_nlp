import torch
from torch import nn


class DnnPosTagger(nn.Module):
    def __init__(self, word_embeddings, hidden_dim, word_vocab_size, tag_vocab_size):
        super(DnnPosTagger, self).__init__()
        emb_dim = word_embeddings.shape[1]
        word_embedding_dim  = word_embeddings.shape[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        # self.word_embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=2, bidirectional=True,
                            batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tag_vocab_size)

    def forward(self, word_idx_tensor):
        embeds = self.word_embedding(word_idx_tensor.to(self.device))  # [batch_size, seq_length, emb_dim]
        lstm_out, _ = self.lstm(embeds.view(embeds.shape[1], 1, -1))  # [seq_length, batch_size, 2*hidden_dim]
        tag_space = self.hidden2tag(lstm_out.view(embeds.shape[1], -1))  # [seq_length, tag_dim]
        tag_scores = F.log_softmax(tag_space, dim=1)  # [seq_length, tag_dim]
        return tag_scores


class DependencyParser(nn.Module):
    def __init__(self, *args):
        super(DependencyParser, self).__init__()
        self.word_embedding =  # Implement embedding layer for words (can be new or pretrained - word2vec/glove)
        self.hidden_dim = self.word_embedding.embedding_dim
        self.encoder =  # Implement BiLSTM module which is fed with word embeddings and outputs hidden representations
        self.edge_scorer =  # Implement a sub-module to calculate the scores for all possible edges in sentence dependency graph
        self.loss_function =  # Implement the loss function described above

    def forward(self, sentence):
        word_idx_tensor, pos_idx_tensor, true_tree_heads = sentence

        # Pass word_idx through their embedding layer

        # Get Bi-LSTM hidden representation for each word in sentence

        # Get score for each possible edge in the parsing graph, construct score matrix

        # Calculate the negative log likelihood loss described above

        return loss, score_mat

from chu_liu_edmonds import decode_mst

def eval_model(model, sentence):
    _, score_mat = model(sentence)
    predicted_tree = decode_mst(score_mat)
    return predicted_tree