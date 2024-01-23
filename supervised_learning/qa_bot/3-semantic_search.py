#!/usr/bin/env python3
"""Semantic Search Module"""
import os
import tensorflow_hub as hub
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def semantic_search(corpus_path, sentence):
    """Performs semantic search on a corpus of documents:

    corpus_path is the path to the corpus of reference documents on which to
    perform semantic search
    sentence is the sentence from which to perform semantic search
    Returns: the reference text of the document most similar to sentence"""

    sentence = [sentence]
    docs = []

    for file in os.listdir(corpus_path):
        if not file.endswith('.md'):
            continue
        with open(corpus_path + "/" + file, "r", encoding='utf-8') as f:
            docs.append(f.read())

    model = hub.load(
        "https://www.kaggle.com/models/google/universal-sentence-encoder/" +
        "frameworks/TensorFlow2/variations/large/versions/2")

    doc_embeddings = model(docs)
    sentence_embedding = model(sentence)

    similarity = cosine_similarity(sentence_embedding, doc_embeddings)
    most_similar = np.argmax(similarity)

    return docs[most_similar]
