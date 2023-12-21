#!/usr/bin/env python3
"""Extract Word2Vec Module"""


def gensim_to_keras(model):
    """converts a gensim word2vec model to a keras Embedding layer:

    model is a trained gensim word2vec models

    Returns: the trainable keras Embedding"""

    return model.wv.get_keras_embedding(True)
