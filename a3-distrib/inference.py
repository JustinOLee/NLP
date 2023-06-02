# inference.py

from cmath import inf
from models import *
from treedata import *
from utils import *
from collections import Counter
from typing import List

import numpy as np


def decode_bad_tagging_model(model: BadTaggingModel, sentence: List[str]) -> List[str]:
    """
    :param sentence: the sequence of words to tag
    :return: the list of tags, which must match the length of the sentence
    """
    pred_tags = []
    for word in sentence:
        if word in model.words_to_tag_counters:
            pred_tags.append(model.words_to_tag_counters[word].most_common(1)[0][0])
        else:
            pred_tags.append("NN") # unks are often NN
    return labeled_sent_from_words_tags(sentence, pred_tags)


def viterbi_decode(model: HmmTaggingModel, sentence: List[str]) -> LabeledSentence:
    """
    :param model: the HmmTaggingModel to use (wraps initial, emission, and transition scores)
    :param sentence: the words to tag
    :return: a LabeledSentence containing the model's predictions. See BadTaggingModel for an example.
    """
    pred_tags = [-1] * len(sentence)
    tag_indexer = model.tag_indexer
    length_no_stop = len(tag_indexer) - 1
    # dp array that keeps track of max score and previous tag
    dp = [[[float('-inf'), -1] for _ in range(len(tag_indexer) - 1)] for _ in sentence]
    for word_idx in range(len(sentence)):
        # if first word, just initial + emission
        if word_idx == 0:
            for tag_index in range(length_no_stop):
                dp[word_idx][tag_index][0] = model.score_init(tag_index) + model.score_emission(sentence, tag_index, word_idx)
        # else get the max prev
        else:
            for tag_index in range(length_no_stop):
                max_prev = [float('-inf'), -1]
                for prev_tag_index in range(length_no_stop):
                    score = model.score_emission(sentence, tag_index, word_idx) + \
                        model.score_transition(prev_tag_index, tag_index) + dp[word_idx - 1][prev_tag_index][0]
                    if score > max_prev[0]:
                        max_prev[0] = score
                        max_prev[1] = prev_tag_index
                dp[word_idx][tag_index][0] = max_prev[0]
                dp[word_idx][tag_index][1] = max_prev[1]

    for word_idx in range(len(sentence) - 1, -1, -1):
        if word_idx == len(sentence) - 1:
            pred_tags[word_idx] = dp[word_idx].index(max(dp[word_idx]))
        else:
            pred_tags[word_idx] = dp[word_idx + 1][pred_tags[word_idx + 1]][1]

    for word_idx in range(len(sentence)):
        pred_tags[word_idx] = tag_indexer.get_object(pred_tags[word_idx])


    return labeled_sent_from_words_tags(sentence, pred_tags)
            


def beam_decode(model: HmmTaggingModel, sentence: List[str], beam_size: int) -> LabeledSentence:
    """
    :param model: the HmmTaggingModel to use (wraps initial, emission, and transition scores)
    :param sentence: the words to tag
    :param beam_size: the beam size to use
    :return: a LabeledSentence containing the model's predictions. See BadTaggingModel for an example.
    """
    beams = [Beam(beam_size) for word in sentence]
    pred_tags = [-1] * len(sentence)
    tag_indexer = model.tag_indexer
    length_no_stop = len(tag_indexer) - 1
    for word_idx in range(len(sentence)):
        # if first word, just initial + emission
        if word_idx == 0:
            for tag_index in range(length_no_stop):
                score = model.score_init(tag_index) + model.score_emission(sentence, tag_index, word_idx)
                beams[word_idx].add((tag_index, -1), score)
        # else get the max prev
        else:
            prev_beam = list(beams[word_idx - 1].get_elts_and_scores())
            for tag_index in range(length_no_stop):
                for prev_beam_elt in prev_beam:
                    prev_tag = prev_beam_elt[0][0]
                    prev_score = prev_beam_elt[1]
                    score = model.score_emission(sentence, tag_index, word_idx) + \
                        model.score_transition(prev_tag, tag_index) + prev_score
                    beams[word_idx].add((tag_index, prev_tag), score)

    for word_idx in range(len(sentence) - 1, -1, -1):
        if word_idx == len(sentence) - 1:
            pred_tags[word_idx] = beams[word_idx].head()[0]
        else:
            next_tag = pred_tags[word_idx + 1]
            next_elts = beams[word_idx + 1].get_elts()
            for elt in next_elts:
                if elt[0] == next_tag:
                    pred_tags[word_idx] = elt[1]
                    break

    for word_idx in range(len(sentence)):
        pred_tags[word_idx] = tag_indexer.get_object(pred_tags[word_idx])


    return labeled_sent_from_words_tags(sentence, pred_tags)
    
