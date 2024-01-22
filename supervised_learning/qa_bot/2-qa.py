#!/usr/bin/env python3
"""Answer Questions Module"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """Finds a snippet of text within a reference document to answer a question
    question is a string containing the question to answer
    reference is a string containing the reference document from which to find
    the answer

    Returns: a string containing the answer
    If no answer is found, return None"""
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load(
        "https://www.kaggle.com/models/seesee/bert/frameworks/TensorFlow2/" +
        "variations/uncased-tf2-qa/versions/1")

    question = tokenizer.tokenize(question)
    paragraph = tokenizer.tokenize(reference)
    tokens = ['[CLS]'] + question + ['[SEP]'] + paragraph + ['[SEP]']

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    input_type_ids = [0] * (1 + len(question) + 1) + [1] * (len(paragraph) + 1)
    input_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0), (input_ids, input_mask,
                                                      input_type_ids))

    outputs = model([input_ids, input_mask, input_type_ids])

    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer


def answer_loop(reference):
    """Answers questions from a reference text:

    reference is the reference text
    If the answer cannot be found in the reference text, respond with
    'Sorry, I do not understand your question.'"""
    while 1:
        Q = input("Q: ")
        if Q.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        else:
            answer = question_answer(Q, reference)
            if not answer:
                print("Sorry, I do not understand your question.")
            else:
                print(f"A: {answer}")
