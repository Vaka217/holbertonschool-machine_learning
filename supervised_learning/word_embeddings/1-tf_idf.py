#!/usr/bin/env python3
"""TF-IDF Module"""
from sklearn.feature_extraction.text import TfidfVectorizer


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

    tfids_vector = TfidfVectorizer(vocabulary=vocab)
    embed = tfids_vector.fit_transform(sentences)
    features = tfids_vector.get_feature_names()
    return embed.toarray(), features
