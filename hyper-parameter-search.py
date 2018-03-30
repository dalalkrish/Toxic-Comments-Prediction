# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:13:36 2018

@author: kdalal
"""
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense
import pickle
import warnings
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
warnings.filterwarnings('ignore')


def data():
    features = pickle.load(open("features.pkl", "rb"))
    labels = pickle.load(open("labels.pkl", "rb"))
    x_train = features[:int(len(features) * 0.8)]
    y_train = labels[:int(len(labels) * 0.8)]
    x_test = features[int(len(features) * 0.8):]
    y_test = labels[int(len(labels) * 0.8):]
    return x_train, y_train, x_test, y_test

def model_creation(x_train, y_train, x_test, y_test):
    
    model = Sequential()
    model.add(Embedding(input_dim = 210338, output_dim={{choice([100, 150, 300, 500])}}, embeddings_initializer='uniform', 
                                     input_length=100))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(LSTM({{choice([150, 256, 512, 1024])}}, return_sequences=True))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(LSTM({{choice([150, 256, 512, 1024])}}, return_sequences=False))
    model.add(Dropout({{uniform(0, 1)}}))
    
    #if conditional({{choice(['three', 'four'])}}) == 'three':        
    #    model.add(LSTM(1024, return_sequences=False))
    #    model.add(Dropout(0.32))
        
    model.add(Dense(6, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})
    print(model.summary())

    model.fit(x_train, y_train, epochs=1, batch_size={{choice([32, 64, 128, 256, 512])}})
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy: ', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model_creation,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    x_train, y_train, x_test, y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
