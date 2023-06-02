from json import encoder
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
from transformer import PositionalEncoding

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)

            

class LanguageModelPredictor(nn.Module):
    def __init__(self, num_positions, d_model, nhead, num_layers, indexer, vocab_size=27, num_classes=27):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=num_positions)
        self.to_model = nn.Linear(num_positions, d_model)
        self.to_classes = nn.Linear(d_model, num_classes)
        self.indexer = indexer
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.PositionalEncoding = PositionalEncoding(d_model, num_positions=num_positions)
        self.num_positions = num_positions

    def forward(self, context):

        indices = []
        for c in context:
            indices.append(self.indexer.index_of(c))

        indices = torch.tensor(indices)
        embeddings = self.embeddings(indices)
        encoded = self.PositionalEncoding(self.to_model(embeddings))
        mask = torch.triu(torch.ones(len(indices), len(indices)) * float('-inf'), diagonal=1)
        transform = self.encoder(encoded, mask)

        return self.log_softmax(self.to_classes(transform))

class NeuralLanguageModel(LanguageModel):
    def __init__(self, num_positions, d_model, nhead, num_layers, indexer):
        self.model = LanguageModelPredictor(num_positions, d_model, nhead, num_layers, indexer)
        self.loss_fcn = nn.NLLLoss()

    def get_next_char_log_probs(self, context):
        log_probs = self.model.forward(context)
        return log_probs[-1].detach().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        chunk_size = self.model.num_positions
        indexer = self.model.indexer
        indices = []
        for c in next_chars:
            indices.append(indexer.index_of(c))

        indices = torch.tensor(indices)

        phrase = context + next_chars
        
        log_probs = torch.zeros(len(phrase), 27)
        chunks = len(phrase) // chunk_size
        ending = 0
        for i in range(chunks):
            log_probs[i * chunk_size : (i + 1) * chunk_size] = self.model.forward(phrase[i * chunk_size : (i + 1) * chunk_size])
            ending = (i + 1) * chunk_size
        if len(phrase[ending:]) > 0:
            log_probs[ending:] = self.model.forward(phrase[ending:])

        loss = 0.0
        
        for i in range(len(phrase) - len(context)):
            loss += log_probs[len(context) - 1 + i][indices[i]]
                
        return loss.item()


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    

    model = NeuralLanguageModel(num_positions=20, d_model=64, nhead=4, num_layers=2, indexer=vocab_index)

    model.model.train()
    model.model.zero_grad()
    optimizer = optim.Adam(model.model.parameters(), lr=1e-3)
    chunk_size = model.model.num_positions
    chunks = len(train_text) // chunk_size

    num_epochs = 7
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        loss_fcn = nn.NLLLoss()
        for chunk in range(chunks - 1):
            text = train_text[chunk * chunk_size : (chunk + 1) * chunk_size]
            context = ' ' + text[0 : chunk_size - 1]
            gold = torch.LongTensor(chunk_size)

            for i in range(chunk_size):
                gold[i] = vocab_index.index_of(text[i])
            log_probs = model.model.forward(context)
            loss = loss_fcn(log_probs, gold)
            model.model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print(t, loss_this_epoch)
    model.model.eval()
    return model
