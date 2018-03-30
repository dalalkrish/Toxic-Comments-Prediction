import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

training_comments = df_train["comment_text"].tolist()
tk = Tokenizer()
tk.fit_on_texts(training_comments)
text2idx = tk.texts_to_sequences(training_comments)
print(text2idx[:7])

vocab_size = len(tk.word_index) + 1 
print("Vocab size: ", vocab_size)

seq_len = 100
features = pad_sequences(text2idx, maxlen=seq_len, dtype='int32')
features[:10, :100]

labels = df_train.iloc[:,-6:].as_matrix()
print(features.shape)
print(labels.shape)

def data(features, labels):
    x_train = features[:int(len(features) * 0.8)]
    y_train = labels[:int(len(labels) * 0.8)]
    x_test = features[int(len(features) * 0.8):]
    y_test = labels[int(len(labels) * 0.8):]
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = data(features, labels)
print("x_train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_test.shape)

def model_creation(x_train, y_train, x_test, y_test, embedding_size = 500, learning_rate=0.05, batch_size = 64, 
                   third_layer=False):
    
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, embeddings_initializer='uniform', 
                                     input_length=seq_len))
    model.add(Dropout(0.45))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.45))
    model.add(LSTM(1024, return_sequences=False))
    model.add(Dropout(0.45))
    
    #if third_layer:
    #    model.add(LSTM(1024, return_sequences=False))
    #    model.add(Dropout(0.32))
        
    model.add(Dense(6, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer="adam")
    print(model.summary())

    model.fit(x_train, y_train, epochs=10, batch_size=batch_size)
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy: ', acc)
    return model

seq_model = model_creation(x_train, y_train, x_test, y_test, third_layer=True)

seq_model.save("lstm_model_03.hd5")
