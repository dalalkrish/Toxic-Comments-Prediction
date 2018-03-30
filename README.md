# Toxic-Comments-Prediction

This is an attempt to solve Kaggle challenge of Toxic Comment Prediction posted by Jigsaw. 

##Application Highlights:

- I used Keras's Sequential rnn model with LSTM cells to make multi class predictions with Tensorflow as backend.

- I used `Hyperas` which is a wrapper of Hyperopt for `hyper-parameter` tuning. Check out file `hyper-parameter-search.py` for implementation details.

- Hyperopt uses _random serach_ and _Tree of Parzen Estimators (TPE)_ for optimizing hyper parameters.

- Even without any data cleaning I was able to achive decent prediciton accuracy by using `Embedding` layer and `LSTM` layers of my rnn architecture.

- I made three submissions and best evaluation accuracy score I got was 0.9654.  

 
