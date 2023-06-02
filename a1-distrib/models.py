# models.py

import math
import random
import numpy as np
from sentiment_data import *
from utils import *
from collections import Counter

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        indicies = []
        for token in sentence:
            token = token.lower()
            idx = -1
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(token)
            else:
                idx = self.indexer.index_of(token)

            indicies.append(idx)
        
        return Counter(indicies)



class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        indicies = []
        for i  in range(len(sentence) - 1):
            firstWord = sentence[i].lower()
            secondWord = sentence[i + 1].lower()
            idx = -1
            if add_to_indexer:
                idx = self.indexer.add_and_get_index((firstWord, secondWord))
            else:
                idx = self.indexer.index_of((firstWord, secondWord))

            indicies.append(idx)
        
        return Counter(indicies)


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        indicies = []
        for token in sentence:
            token = token.lower()
            idx = -1
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(token)
            else:
                idx = self.indexer.index_of(token)

            if idx not in indicies:
                indicies.append(idx)
        
        return Counter(indicies)


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, featurizer: FeatureExtractor):
        self.weights = weights
        self.featurizer = featurizer
    
    def predict(self, sentence: List[str]) -> int:
        counter = self.featurizer.extract_features(sentence)
        total_weight = 0

        for token in counter:
            total_weight += counter[token] * self.weights[token]

        if total_weight > 0:
            return 1
        return 0
        


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights: np.ndarray, featurizer: FeatureExtractor):
        self.weights = weights
        self.featurizer = featurizer
    
    def predict(self, sentence: List[str]) -> int:
        counter = self.featurizer.extract_features(sentence)
        total_weight = 0

        for token in counter:
            total_weight += counter[token] * self.weights[token]

        logs = math.exp(total_weight)

        prob = logs / (1 + logs)

        if prob > 0.5:
            return 1
        return 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    training_vectors = []
    for ex in train_exs:
        training_vectors.append((feat_extractor.extract_features(sentence=ex.words, add_to_indexer=True), ex.label))

    indexer = feat_extractor.get_indexer()
    weights = np.zeros(len(indexer))
    
    epochs = 20 
    for epoch in range(1, epochs + 1):
        lr = 1/epoch
        random.shuffle(training_vectors)
        for ex in training_vectors:
            pred = 0
            features = ex[0]
            total_weight = 0

            for feature in features:
                total_weight += features[feature] * weights[feature]

            if total_weight > 0:
                pred = 1
            
            if pred != ex[1]:
                if ex[1] == 1:
                    for feature in features:
                        weights[feature] += lr * features[feature]
                else:
                    for feature in features:
                        weights[feature] -= lr * features[feature]
    
    return PerceptronClassifier(weights=weights, featurizer=feat_extractor)


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    training_vectors = []
    for ex in train_exs:
        training_vectors.append((feat_extractor.extract_features(sentence=ex.words, add_to_indexer=True), ex.label))

    indexer = feat_extractor.get_indexer()
    weights = np.zeros(len(indexer))
    
    epochs = 20 
    for epoch in range(1, epochs + 1):  
        lr = 1/epoch
        random.shuffle(training_vectors)
        for ex in training_vectors:
            pred = 0
            features = ex[0]
            total_weight = 0

            for feature in features:
                total_weight += features[feature] * weights[feature]

            logs = math.exp(total_weight)
            prob = logs / (1 + logs)
            
            if prob > 0.5:
                pred = 1
            
            if pred == 1:
                if ex[1] == 1:
                    for feature in features:
                        weights[feature] += lr * features[feature] * (1 - prob)
                else:
                    for feature in features:
                        weights[feature] -= lr * features[feature] * (1 - prob)
            else:
                if ex[1] == 1:
                    for feature in features:
                        weights[feature] += lr * features[feature] * prob
                else:
                    for feature in features:
                        weights[feature] -= lr * features[feature] * prob

    return LogisticRegressionClassifier(weights=weights, featurizer=feat_extractor)
    


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
