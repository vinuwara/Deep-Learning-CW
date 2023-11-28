# Deep Learning Models for Yelp Reviews Analysis

This repository contains the implementation and comparison of two deep learning architectures applied to sentiment analysis using the Yelp Reviews dataset.

## Group Members

* Vinuwara Ronath (IIT ID - 20210167 / RGU ID - 2119942)
* Ridmika Hasaranga (IIT ID - 20210763 / RGU ID - 2117527)
* Ishan Fernando (IIT ID - 20210549 / RGU ID - 2117523)

## Introduction

This coursework aims to address a real-world problem of sentiment analysis on Yelp Reviews using two distinct deep learning architectures. The models have been trained, validated, and tested on the Yelp Reviews dataset, accessible from [Yelp Dataset](https://www.yelp.com/dataset).

## Deep Learning Architectures

### Model 1: [CNN]

Sentiment analysis is a crucial natural language processing (NLP) task for determining the sentiment expressed in text. This report details the methodology, experimental results, evaluation criteria, limitations, and potential future enhancements for a sentiment analysis model implemented using a Convolutional Neural Network (CNN).

### Model 2: [RNN]

Sentiment analysis is a natural language processing (NLP) task aimed at determining the sentiment expressed in a piece of text. This report outlines the methodology, experimental results, evaluation criteria, limitations, and potential future enhancements for a sentiment analysis model implemented using a Recurrent Neural Network (RNN).

## Repository Structure

├───CNN-model <br>
│   └───.ipynb_checkpoints <br>
│   └───CNN model.ipynb <br>
│   └───CNN_model_final.h5 <br>
└───RNN-model <br>
&nbsp;&nbsp;&nbsp;└───.ipynb_checkpoints  <br>
&nbsp;&nbsp;&nbsp;└───RNN model.ipynb <br>
&nbsp;&nbsp;&nbsp;└───RNN_model_final.h5 <br>
├── README.md
	
## Acknowledgments

### Libraries
* pandas
* numpy
* sklearn.model_selection import train_test_split
* sklearn.metrics import confusion_matrix, classification_report
* matplotlib.pyplot
* seaborn
* tensorflow.keras.preprocessing.text import Tokenizer
* tensorflow.keras.preprocessing.sequence import pad_sequences
* tensorflow.keras.models import Sequential, load_model
* tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM
* tensorflow.keras.optimizers import Adam
* os
* pickle
