"""
This file contains the implementation of the Continuous Bag of Words (CBOW) model.
"""

import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

class CBOW:
    """
    Class for the Continuous Bag of Words (CBOW) model.
    Note that the vocabulary should be provided.
    """
    def __init__(self,corpus,vocab,window_size=10,embedding_dim=100):
        self.corpus = corpus #Document iterator (Data frame or list of documents for exemple)
        self.vocab_indexed = {word: i for i, word in enumerate(vocab)} #Vocabulary indexed
        self.window_size = window_size #Size of the context window
        self.embedding_dim = embedding_dim #Dimension of the word embeddings
        self.vocab_size = len(vocab) #Size of the vocabulary
        self.model = None #Resulting model

    #Get training data
    def get_context(self):
        """
        Method to get the context and target words for training the model using the corpus.
        """
        X = []
        y = []
        k = self.window_size
        for doc in self.corpus:
            for i,w in enumerate(doc):
                #Get the context and target words
                context = [self.vocab_indexed[doc[j]] for j in range(max(0,i-k),min(len(doc),i+k+1)) if i!=j]
                X.append(context)
                y.append(self.vocab_indexed[w])

            #0-padding
            for i in range(k):
                X[-len(doc)+i]=[0]*(k-i)+X[-len(doc)+i]
                X[-i-1]=X[-i-1]+[0]*(k-i)

        #Generate one-hot vectors to feed the model
        for j,context in enumerate(X):
            X_onehot = np.zeros((self.vocab_size,2*self.window_size))
            for i,word_index in enumerate(context):
                X_onehot[word_index][i] = 1
            y_onehot = np.zeros(self.vocab_size)
            y_onehot[y[j]] = 1
            yield X_onehot, y_onehot

    #Train the model
    def train(self,batch_size=10,epochs=10):
        """
        Method to train the model using the corpus.
        Inputs :
            - batch_size : size of the batch
            - epochs : number of epochs
        Outputs :
            - None
        """

        # Get the generator for the training data
        generator = self.get_context()

        # Determine the number of steps per epoch
        total_data_points = sum(len(doc) for doc in self.corpus)
        steps_per_epoch = total_data_points // batch_size

        # Create the model
        model = Sequential(
            [Embedding(input_dim = self.vocab_size, output_dim = self.embedding_dim, input_length=2*self.window_size),
            GlobalAveragePooling1D(),
            Dense(self.vocab_size, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs)
        self.model = model

    #Get word embeddings
    