import json
import string
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

nltk.download("punkt")
nltk.download("wordnet")


# used a dictionary to represent an intents JSON file
data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "Hello",
                "How are you?",
                "Hi there",
                "Hi",
                "Whats up"],
            "responses": [
                "Howdy Partner!",
                "Hello",
                "How are you doing?",
                "Greetings!",
                "How do you do?",
            ],
        },
        {
            "tag": "age",
            "patterns": [
                "how old are you?",
                "when is your birthday?",
                "when was you born?",
            ],
            "responses": [
                "I am 24 years old",
                "I was born in 1996",
                "My birthday is July 3rd and I was born in 1996",
                "03/07/1996",
            ],
        },
        {
            "tag": "date",
            "patterns": [
                "what are you doing this weekend?",
                "do you want to hang out some time?",
                "what are your plans for this week",
            ],
            "responses": [
                "I am available all week",
                "I don't have any plans",
                "I am not busy",
            ],
        },
        {
            "tag": "name",
            "patterns": [
                "what's your name?",
                "what are you called?",
                "who are you?"],
            "responses": ["My name is Kippi", "I'm Kippi", "Kippi"],
        },
        {
            "tag": "goodbye",
            "patterns": ["bye", "g2g", "see ya", "adios", "cya"],
            "responses": [
                "It was nice speaking to you",
                "See you later",
                "Speak soon!",
            ],
        },
    ]
}

# initializing lemmatizer to get stem of words
lemmatizer = WordNetLemmatizer()
# Each list to create
words = []
classes = []
doc_X = []
doc_y = []
# Loop through all the intents
# tokenize each pattern and append tokens to words, the patterns and
# the associated tag to their associated list
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])

    # add the tag to the classes if it's not there already
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
# lemmatize all the words in the vocab and convert them to lowercase
# if the words don't appear in punctuation
words = [
    lemmatizer.lemmatize(word.lower())
    for word in words
    if word not in string.punctuation
]
# sorting the vocab and classes in alphabetical order and taking the
# # set to ensure no duplicates occur
words = sorted(set(words))
classes = sorted(set(classes))


# list for training data
training = []
out_empty = [0] * len(classes)
# creating the bag of words model
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    # mark the index of class that the current pattern is associated
    # to
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    # add the one hot encoded BoW and associated classes to training 
    training.append([bow, output_row])
# shuffle the data and convert it to an array
random.shuffle(training)
training = np.array(training, dtype=object)
# split the features and target labels
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# defining some parameters
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200
# the deep learning model
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation = "softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=["accuracy"])
print(model.summary())
model.fit(x=train_X, y=train_y, epochs=200, verbose=1)


# defining some parameters
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200
# the deep learning model
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation = "softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=["accuracy"])
print(model.summary())
model.fit(x=train_X, y=train_y, epochs=200, verbose=1)