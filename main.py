# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
import re


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


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
            #print(line_set)
            tag = line_set[6]
            word_index = line_set[0]
            word = line_set[1]
            word_pos = line_set[3]
            X_representation = [word_index,word, word_pos]
            if len(list_of_sentences_with_tags) <= sentence_index:
                list_of_sentences_with_tags.append([(X_representation, tag)])
                list_of_sentences.append([X_representation])

            else:
                list_of_sentences_with_tags[sentence_index].append((X_representation, tag))
                list_of_sentences[sentence_index].append(X_representation)
    return list_of_sentences, list_of_sentences_with_tags



def create_data(train_path,test_path, com_path):
    list_of_sentences_train, list_of_sentences_with_tags_train = preprocess(train_path)
    list_of_sentences_test, list_of_sentences_with_tags_test = preprocess(test_path)
    list_of_sentences_comp,_ = preprocess(com_path)
    print("pp")




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    train_path ='train.labeled'
    test_path = 'test.labeled'
    com_path = 'comp.unlabeled'
    create_data(train_path,test_path, com_path)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
