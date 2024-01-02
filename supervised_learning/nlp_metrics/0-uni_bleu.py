#!/usr/bin/env python3
"""Unigram BLEU Score"""
from math import exp


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence:

    references is a list of reference translations
    each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence

    Returns: the unigram BLEU score"""

    return max([sum(match in reference for match in sentence)
                for reference in references]) / len(sentence)
