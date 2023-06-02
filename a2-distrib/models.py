# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class FFNN(nn.Module):
    """
    Adapted from ffnn_example.py

    The forward() function does the important computation. The backward() method is inherited from nn.Module and
    handles backpropagation.
    """
    def __init__(self, inp, hid, out):
        """
        Constructs the computation graph by instantiating the various layers and initializing weights.

        :param inp: size of input (integer)
        :param hid: size of hidden layer(integer)
        :param out: size of output (integer), which should be the number of classes
        """
        super(FFNN, self).__init__()
        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """ 
        return self.log_softmax(self.W(self.g(self.V(x))))


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, ffnn: FFNN, embedding_layer, indexer: Indexer):
        self.ffnn = ffnn
        self.embedding_layer = embedding_layer
        self.indexer = indexer
    
    def predict(self, ex_words: List[str]) -> int:
        input = []
        for word in ex_words:
            index = self.indexer.index_of(word)
            if index == -1:
                index = self.indexer.index_of("UNK")
            word_tensor = torch.tensor(index)
            input.append(self.embedding_layer(word_tensor))
        x = torch.stack(input)
        avg = torch.mean(x, 0)
        log_probs = self.ffnn.forward(avg)
        return torch.argmax(log_probs)



def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    # Define some constants
    # 50 or 300, depending on the dataset being used
    feat_vec_size = 300
    # Let's use 4 hidden units
    embedding_size = args.hidden_size
    # We're using 2 classes. 
    num_classes = 2
    batch_size = args.batch_size

    # Get the data in tensors
    embedding_layer = word_embeddings.get_initialized_embedding_layer()
    indexer = word_embeddings.word_indexer

    # words list, just for readability
    sentences = [ex.words for ex in train_exs]
    # Get the actual nn input by passing the index through the embedding layer
    train_inputs = []

    for sentence in sentences:
        embeddings = []
        for word in sentence:
            index = indexer.index_of(word)
            if index == -1:
                index = indexer.index_of("UNK")
            word_tensor = torch.tensor(index)
            embeddings.append(embedding_layer(word_tensor))

        train_inputs.append(embeddings)

    gold_labels = [ex.label for ex in train_exs]

    # RUN TRAINING AND TEST
    num_epochs = args.num_epochs
    ffnn = FFNN(feat_vec_size, embedding_size, num_classes)
    initial_learning_rate = args.lr
    optimizer = optim.Adam(ffnn.parameters(), lr=initial_learning_rate)
    loss = nn.NLLLoss()

    if batch_size == 1:
        for epoch in range(0, num_epochs):
            ex_indices = [i for i in range(0, len(train_inputs))]
            random.shuffle(ex_indices)
            total_loss = 0.0
            for idx in ex_indices:
                x = torch.stack(train_inputs[idx])
                avg = torch.mean(x, 0)
                y = torch.tensor(gold_labels[idx])
                # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
                ffnn.zero_grad()
                log_probs = ffnn.forward(avg)
                # Use built-in NLLLoss
                output = loss(log_probs, y)
                total_loss += output
                # Computes the gradient and takes the optimizer step
                output.backward()
                optimizer.step()
            print("Total loss on epoch %i: %f" % (epoch, total_loss))
    else:    
        for epoch in range(0, num_epochs):
            ex_indices = [i for i in range(0, len(train_inputs))]
            random.shuffle(ex_indices)
            total_loss = 0.0
            x_batch = []
            y_batch = []
            counter = 0
            for idx in ex_indices:
                counter += 1
                x = torch.stack(train_inputs[idx])
                avg = torch.mean(x, 0).tolist()
                x_batch.append(avg)
                y_batch.append(gold_labels[idx])
                if len(x_batch) == batch_size or counter == len(ex_indices):
                    # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
                    ffnn.zero_grad()
                    log_probs = ffnn.forward(torch.tensor(x_batch))
                    # Use built-in NLLLoss
                    output = loss(log_probs, torch.tensor(y_batch))
                    total_loss += output
                    # Computes the gradient and takes the optimizer step
                    output.backward()
                    optimizer.step()
                    x_batch.clear()
                    y_batch.clear()

            print("Total loss on epoch %i: %f" % (epoch, total_loss))
    
    return NeuralSentimentClassifier(ffnn, embedding_layer, indexer)
