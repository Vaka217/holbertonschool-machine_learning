#!/usr/bin/env python3
"""TF-IDF Module"""
import numpy as np
import math
import re


def tf_idf(sentences, vocab=None):
    """Creates a TF-IDF embedding:

    sentences is a list of sentences to analyze
    vocab is a list of the vocabulary words to use for the analysis
    If None, all words within sentences should be used

    Returns: embeddings, features
    embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
    s is the number of sentences in sentences
    f is the number of features analyzed
    features is a list of the features used for embeddings"""

    if vocab is None:
        vocab = []
        for sentence in sentences:
            vocab.extend(re.sub(r"\b\w{1}\b", "", re.sub(
                r"[^a-zA-Z0-9\s]", " ", sentence.lower())).split())
        vocab = sorted(list(set(vocab)))

    embeddings = np.zeros((len(sentences), len(vocab)))

    for i, sentence in enumerate(sentences):
        words = sentence.split()
        for word in words:
            word = re.sub(r"\b\w{1}\b", "", re.sub(
                r"[^a-zA-Z0-9\s]", " ", word.lower())).strip()
            if word in vocab:
                embeddings[i][vocab.index(word)] += 1
        if sum(embeddings[i]) > 0:
            embeddings[i] /= sum(embeddings[i])

    logs = np.ones((len(embeddings.T)))

    for i, num_words in enumerate(embeddings.T):
        if np.count_nonzero(num_words) > 0:
            print(math.log(len(sentences) / np.count_nonzero(num_words), 10))
            logs[i] = math.log(
                len(sentences) / np.count_nonzero(num_words), 10)
            print(logs[i])

    return embeddings * logs, vocab
