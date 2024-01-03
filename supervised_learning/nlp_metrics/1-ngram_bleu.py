#!/usr/bin/env python3
"""N-gram BLEU Score"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence:

    references is a list of reference translations
    each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    n is the size of the n-gram to use for evaluation

    Returns: the n-gram BLEU score"""

    BP = min(1, np.exp(1 - len(min(references, key=len)) / len(sentence)))
    n_grams = []
    n_grams_ref = 0

    for reference in references:
        for i in range(len(sentence) - (n - 1)):
            n_grams_ref += 1 if any(sentence[i:i + n] == reference[j:j+n]
                                    for j in range(len(reference) - (n - 1)))\
                else 0
        n_grams.append(n_grams_ref)

    precision = max(n_grams) / (i + 1)

    return BP * np.exp(np.log(precision))
