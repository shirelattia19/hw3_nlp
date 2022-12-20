import os.path
import re
import numpy as np
from gensim.models import Word2Vec
import pickle



def preprocess(path):
    sentence_index = 0
    list_of_sentences = []
    list_of_sentences_with_tags = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line == "\n":
                sentence_index += 1
                continue

            if line[-1:] == "\n":
                line = line[:-1]
            line_set = re.split(r'\t+', line)
            # print(line_set)
            tag = line_set[6]
            word_index = line_set[0]
            word = line_set[1]
            word_pos = line_set[3]
            X_representation = [word_index, word, word_pos, tag]
            if len(list_of_sentences_with_tags) <= sentence_index:
                list_of_sentences_with_tags.append([X_representation])
            else:
                list_of_sentences_with_tags[sentence_index].append(X_representation)
    list_of_sentences_with_tags = [np.array(sen).T.tolist() for sen in list_of_sentences_with_tags]
    return list_of_sentences_with_tags


def create_data(train_path, test_path, com_path):
    list_of_sentences_with_tags_train = preprocess(train_path)
    with open(f"train.data", 'wb+') as f:
        pickle.dump(list_of_sentences_with_tags_train, f)
    list_of_sentences_with_tags_test = preprocess(test_path)
    with open(f"test.data", 'wb+') as f:
        pickle.dump(list_of_sentences_with_tags_test, f)
    list_of_sentences_comp = preprocess(com_path)
    with open(f"comp.data", 'wb+') as f:
        pickle.dump(list_of_sentences_comp, f)


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
    sentences = [sen[1] for sen in (
                list_of_sentences_with_tags_train + list_of_sentences_with_tags_test + list_of_sentences_with_tags_comp)]
    trained_model = Word2Vec(sentences=sentences, vector_size=100, window=2, min_count=1, workers=4, epochs=100,
                             seed=42)
    trained_model.save("trained_word2vec.model")





if __name__ == '__main__':
    # # Preprocess the data files
    # train_path ='train.labeled'
    # test_path = 'test.labeled'
    # com_path = 'comp.unlabeled'
    # create_data(train_path,test_path, com_path)

    # # Create the embedding models
    # pre_embedding()


