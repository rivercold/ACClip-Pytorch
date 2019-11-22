import os
import sys
import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe


def load_dataset(dataset="imdb", batch_size=32, word_dim=300):
    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.

    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.

    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    """

    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=False, batch_first=True,
                      fix_length=200)
    LABEL = data.LabelField(sequential=False)
    if dataset == "imdb":
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    elif dataset == "sst":
        train_data, dev_data, test_data = datasets.SST.splits(TEXT, LABEL,
                                           fine_grained=False)
    print ("after split, we start to build vocab")
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=word_dim), min_freq=10)
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    train_data, valid_data = train_data.split()  # Further splitting of train & validation
    print ("train: {}; dev: {}; test: {}".format(len(train_data), len(valid_data), len(test_data)))
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                   batch_size=batch_size,
                                                                   sort_key=lambda x: len(x.text), repeat=False,
                                                                   shuffle=True)

    '''Alternatively we can also use the default configurations'''
    # train_iter, test_iter = datasets.IMDB.iters(batch_size=32)
    vocab_size = len(TEXT.vocab)
    class_num = len(LABEL.vocab)

    return train_iter, valid_iter, test_iter, class_num, vocab_size, word_embeddings