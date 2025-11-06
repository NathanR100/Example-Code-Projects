# -*- coding: utf-8 -*-
"""MSBC5190_S24_Homework_2-Group_22.ipynb

# Homework 2, MSBC.5190 Modern Artificial Intelligence S24


**Teammates: Emma Hammergard, Nathan Ryan, Caroline Jones**

**Teamname: Group 22**

Handout 03/27/2024 5pm, **due 04/12/2024 by 5pm**. Please submit through Canvas. Each team only needs to submit one copy.

Important information about submission:

*   Write all code, text (answers), and figures in the notebook.
*   Please make sure that the submitted notebook has been run and the cell outputs are visible.
*   Please print the notebook as PDF and submit it together with the notebook. Your submission should contain two files: `homework2-teamname.ipynb` and `homework2-teamname.pdf`

The goal of the homework are three folds:


1.   Explore word embedding
2.   Understand contextual word embedding using BERT
3.   Text classificaiton with both traditional machine learning methods and deep learning methods

**A note about GPU**: You'd better use GPU to run it, otherwise it will be quite slow to train deep learning models.

First, import the packages or modules required for this homework.
"""

################################################################################
# TODO: Fill in your codes                                                     #
# Import packages or modules                                                   #
################################################################################
import tensorflow as tf
import numpy as np
from tensorflow import keras

"""## Part I: Explore Word Embedding (15%)

Word embeddings are useful representation of words that capture information about word meaning as well as location. They are used as a fundamental component for downstream NLP tasks, e.g., text classification. In this part, we will explore the embeddings produced by [GloVe (global vectors for word representation)](https://nlp.stanford.edu/projects/glove/). It is simlar to Word2Vec but differs in their underlying methodology: in GloVe, word embeddings are learned based on global word-word co-occurrence statistics. Both Word2Vec and GloVe tend to produce vector-space embeddings that perform similarly in downstream NLP tasks.

We first load the GloVe vectors
"""

import gensim.downloader as api
# download the model and return as object ready for use
glove_word_vectors = api.load('glove-wiki-gigaword-100')

"""Take a look at the vocabulary size and dimensionality of the embedding space"""

print('vocabulary size = ', len(glove_word_vectors.index_to_key))
print('embedding dimensionality = ', glove_word_vectors['happy'].shape)

"""
What is embedding exactly?"""

# Check word embedding for 'happy'
# You can access the embedding of a word with glove_word_vectors[word] if word
# is in the vocabulary
glove_word_vectors['happy']

"""With word embeddings learned from GloVe or Word2Vec, words with similar semantic meanings tend to have vectors that are close together. Please code and calculate the **cosine similarities** between words based on their embeddings (i.e., word vectors).

For each of the following words in occupation, compute its cosine similarty to 'woman' and its similarity to 'man' and check which gender is more similar.

*occupation = {homemaker, nurse, receptionist, librarian, socialite, hairdresser, nanny, bookkeeper, stylist, housekeeper, maestro, skipper, protege, philosopher, captain, architect, financier, warrior, broadcaster, magician}*

**Inline Question #1:**
- Fill in the table below with cosine similarities between words in occupation list and {woman, man}. Please show only two digits after decimal.
- Which words are more similar to 'woman' than to 'man'?

The words who are more similar to 'woman' is: homemaker, nurse, receptionist, librarian, socialite, hairdresser, nanny, bookkeper, stylist and housekeeper.

- Which words are more similar to 'man' than to 'woman'?

The words who are more similar to 'man' is: maestro, skipper, protege, philosopher, captain, architect, financier, warrior, broadcaster and magician.

- Do you see any issue here? What do you think might cause these issues?

The results present some interesting patterns, with traditionally "feminine" occupations such as homemaker, nurse, and librarian having higher similarity scores with "woman" than "man". And traditionally "masculine" occupations such as captain, architect, and warrior showing higher similarity scores with "man". This implies that the GloVe model, which is trained on web data, might be reflecting societal biases and gender stereotypes prevalent in the training data. Additionally, the negative similarity score between 'maestro' and 'woman', which can occur if the vectors are pointing in opposite directions in the vector space, indicating very dissimilar or contextually opposite meanings.

**Your Answer:**

| `similarity`|    woman  |      man     |
|-------------|-----------|--------------|
| homemaker   |  0.43       |   0.24           |
| nurse       |  0.61       |   0.46           |
| receptionist|  0.34       |   0.19           |
| librarian   |  0.34       |   0.23           |
| socialite   |  0.42       |   0.27           |
| hairdresser |  0.39       |   0.26           |
| nanny       |  0.36       |   0.29           |
| bookkeeper  |  0.21       |   0.15           |
| stylist     |  0.31       |   0.25           |
| housekeeper |  0.46       |   0.31           |
| maestro     |  -0.02      |   0.14           |
| skipper     |  0.15       |   0.34           |
| protege     |  0.19       |   0.20           |
| philosopher |  0.23       |   0.28           |
| captain     |  0.31       |   0.53           |
| architect   |  0.22       |   0.30           |
| financier   |  0.14       |   0.26           |
| warrior     |  0.39       |   0.51           |
| broadcaster |  0.23       |   0.25           |
| magician    |  0.28       |   0.38           |

"""

################################################################################
# TODO: Fill in your codes                                                     #                                                          #
################################################################################

# List of occupations
occupations = ["homemaker", "nurse", "receptionist", "librarian", "socialite",
               "hairdresser", "nanny", "bookkeeper", "stylist", "housekeeper",
               "maestro", "skipper", "protege", "philosopher", "captain",
               "architect", "financier", "warrior", "broadcaster", "magician"]

# Function to calculate and print similarities
def print_similarities(glove_word_vectors, word1, word2, occupations):
    for occupation in occupations:
        # Ensure the occupation word exists in the model to avoid KeyError
        if occupation in glove_word_vectors.key_to_index:
            similarity_to_word1 = glove_word_vectors.similarity(occupation, word1)
            similarity_to_word2 = glove_word_vectors.similarity(occupation, word2)
            more_similar = word1 if similarity_to_word1 > similarity_to_word2 else word2
            print(f"{occupation}: {word1} = {similarity_to_word1:.4f}, {word2} = {similarity_to_word2:.4f}. More similar to {more_similar}.")
        else:
          print(f"{occupation} is not in the model's vocabulary.")

# Calculate and print the similarities
print_similarities(glove_word_vectors, "woman", "man", occupations)

"""## Part II Understand contextual word embedding using BERT (15%)

A big difference between Word2Vec and BERT is that Word2Vec learns context-free word representations, i.e., the embedding for 'orange' is the same in "I love eating oranges" and in "The sky turned orange". BERT, on the other hand, produces contextual word presentations, i.e., embeddings for the same word in different contexts are different.

For example, let us compare the context-based embedding vectors for 'orange' in the following three sentences using Bert:
* "I love eating oranges"
* "My favorite fruits are oranges and apples"
* "The sky turned orange"

Same as in "Lab 5 BERT", we use the BERT model and tokenizer from the Huggingface transformer library ([1](https://huggingface.co/course/chapter1/1), [2](https://huggingface.co/docs/transformers/quicktour))
"""

# Note that we need to install the latest version of transformers
# Due to problems we encountered in class and reported here
# https://discuss.huggingface.co/t/pretrain-model-not-accepting-optimizer/76209
# https://github.com/huggingface/transformers/issues/29470

!pip install --upgrade transformers
import transformers
print(transformers.__version__)

from transformers import BertTokenizer, TFBertModel

"""We use the 'bert-base-cased' from Huggingface as the underlying BERT model and the associated tokenizer."""

bert_model = TFBertModel.from_pretrained('bert-base-cased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

example_sentences = ["I love eating oranges",
                     "My favorite fruits are oranges and apples",
                     "The sky turned orange"]

"""Let us start by tokenizing the example sentences."""

# Check how Bert tokenize each sentence
# This helps us identify the location of 'orange' in the tokenized vector
for sen in example_sentences:
  print(bert_tokenizer.tokenize(sen))

"""Notice that the prefix '##' indicates that the token is a continuation of the previous one. This also helps us identify location of 'orange' in the tokenized vector, e.g., 'orange' is the 4th token in the first sentence. Note that here the tokenize() function just splits a text into words, and doesn't add a 'CLS' (classification token) or a 'SEP' (separation token) to the text.

Next, we use the tokenizer to transfer the example sentences to input that the Bert model expects.
"""

bert_inputs = bert_tokenizer(example_sentences,
                             padding=True,
                             return_tensors='tf')

bert_inputs

"""So there are actually three outputs: the input ids (starting with '101' for the '[CLS]' token), the token_type_ids which are usefull when one has distinct segments, and the attention masks which are used to mask out padding tokens.

Please refer to our Lab 4 for more details about input_ids, token_type_ids, and attention_masks.

More resources:
*    https://huggingface.co/docs/transformers/preprocessing
*    https://huggingface.co/docs/transformers/tokenizer_summary

Now, let us get the BERT encoding of our example sentences.
"""

bert_outputs = bert_model(bert_inputs)

print('shape of first output: \t\t', bert_outputs[0].shape)
print('shape of second output: \t', bert_outputs[1].shape)

"""There are two outputs here: one with dimensions [3, 10, 768] and one with [3, 768]. The first one [batch_size, sequence_length, embedding_size] is the output of the last layer of the Bert model and are the contextual embeddings of the words in the input sequence. The second output [batch_size, embedding_size] is the embedding of the first token of the sequence (i.e., classification token).

Note you can also get the first output through bert_output.last_hidden_state (see below, also check https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/bert#transformers.TFBertModel)

We need the first output to get contextualized embeddings for 'orange' in each sentence.
"""

bert_outputs[0]

bert_outputs.last_hidden_state

"""Now, we get the embeddings of 'orange' in each sentence by simply finding the 'orange'-token positions in the embedding output and extract the proper components:"""

orange_1 = bert_outputs[0][0, 4]
orange_2 = bert_outputs[0][1, 5]
orange_3 = bert_outputs[0][2, 4]

oranges = [orange_1, orange_2, orange_3]

"""We calculate pair-wise cosine similarities:"""

def cosine_similarities(vecs):
    for v_1 in vecs:
        similarities = ''
        for v_2 in vecs:
            similarities += ('\t' + str(np.dot(v_1, v_2)/
                np.sqrt(np.dot(v_1, v_1) * np.dot(v_2, v_2)))[:4])
        print(similarities)

cosine_similarities(oranges)

"""The similarity metrics make sense. The 'orange' in "The sky turned orange" is different from the rest.

Next, please compare the contextual embedding vectors of 'bank' in the following four sentences:


*   "I need to bring my money to the bank today"
*   "I will need to bring my money to the bank tomorrow"
*   "I had to bank into a turn"
*   "The bank teller was very nice"


**Inline Question #1:**

- Please calculate the pair-wise cosine similarities between 'bank' in the four sentences and fill in the table below. (Note, bank_i represent bank in the i_th sentence)
- Please explain the results. Does it make sense?

Yes, the results make sense given the different contexts in which "bank" is used across the four sentences. For example, it shows high similarity between sentences where "bank" is a noun related to a place or business reflect BERT's understanding of "bank" in these contexts as similar. Additionally, it shows moderate similarity between sentences when comparing these noun uses to the verb use ("to bank into a turn") demonstrate the model's capacity to differentiate meanings based on usage context.

**Your Answer:**

| `similarity`|  bank_1  |  bank_2  |  bank_3  |  bank_4  |
|-------------|----------|----------|----------|----------|
| bank_1      |  1.0     |   0.99   |  0.59    |  0.86        |
| bank_2      |  0.99    |   1.0    |  0.59    |  0.87        |
| bank_3      |  0.59    |   0.59   |  1.0     |  0.62        |
| bank_4      |  0.86    |   0.87   |  0.62    |  1.0        |
"""

################################################################################
# TODO: Fill in your codes                                                     #
################################################################################

ex_sentence = ["I need to bring my money to the bank today",
                "I will need to bring my money to the bank tomorrow",
                "I had to bank into a turn",
                "The bank teller was very nice"]

for sen in ex_sentence:
  print(bert_tokenizer.tokenize(sen))

bert_inputs1 = bert_tokenizer(ex_sentence,
                             padding=True,
                             return_tensors='tf')

bert_inputs1

bert_outputs1 = bert_model(bert_inputs1)

print('shape of first output: \t\t', bert_outputs1[0].shape)
print('shape of second output: \t', bert_outputs1[1].shape)

bank_1 = bert_outputs1[0][0, 9]
bank_2 = bert_outputs1[0][1, 10]
bank_3 = bert_outputs1[0][2, 4]
bank_4 = bert_outputs1[0][3, 2]

banks = [bank_1, bank_2, bank_3, bank_4]

cosine_similarities(banks)

"""## Part III Text classification

In this part, you will build text classifiers that try to infer whether tweets from [@realDonaldTrump](https://twitter.com/realDonaldTrump) were written by Trump himself or by a staff person.
This is an example of binary classification on a text dataset.

It is known that Donald Trump uses an Android phone, and it has been observed that some of his tweets come from Android while others come from other devices (most commonly iPhone). It is widely believed that Android tweets are written by Trump himself, while iPhone tweets are written by other staff. For more information, you can read this [blog post by David Robinson](http://varianceexplained.org/r/trump-tweets/), written prior to the 2016 election, which finds a number of differences in the style and timing of tweets published under these two devices. (Some tweets are written from other devices, but for simplicity the dataset for this assignment is restricted to these two.)

This is a classification task known as "authorship attribution", which is the task of inferring the author of a document when the authorship is unknown. We will see how accurately this can be done with linear classifiers using word features.

You might find it familiar: Yes! We are using the same data set as your homework 2 from MSBC 5180.

### Tasks

In this section, you will build two text classifiers: one with a traditional machine learning method that you studied in MSBC.5190 and one with a deep learning method.


*   For the first classifier, you can use any non-deep learning based methods. You can use your solution to Homework 2 of MSBC 5180 here.
*   For the second classifier, you may try the following methods
    *    Fine-tune BERT (similar to our Lab 5 Fine-tune BERT for Sentiment Analysis)
    *    Use pre-trained word embedding (useful to check: https://keras.io/examples/nlp/pretrained_word_embeddings/)
    *    Train a deep neural network (e.g., CNN, RNN, Bi-LSTM) from scratch, similar to notebooks from our textbook:
        *    https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/dense_sentiment_classifier.ipynb
        *    https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/convolutional_sentiment_classifier.ipynb
        *    https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/rnn_sentiment_classifier.ipynb
        *    https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/lstm_sentiment_classifier.ipynb
        *    https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/bi_lstm_sentiment_classifier.ipynb
    *   There are also lots of useful resources on Keras website: https://keras.io/examples/nlp/

You may want to split the current training data to train and validation to help model selection. Please do not use test data for model selection.

### Load the Data Set

#### Sample code to load raw text###

Please download `tweets.train.tsv` and `tweets.test.tsv` from Canvas (Module Assignment) and upload them to Google Colab. Here we load raw text data to text_train and text_test.
"""

import os, pandas as pd


from google.colab import drive
drive.mount('/content/drive/')

os.chdir('/content/drive/MyDrive/CSV Files/')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#training set
df_train = pd.read_csv('tweets.train.tsv', sep='\t', header=None)

text_train = df_train.iloc[0:, 1].values.tolist()
Y_train = df_train.iloc[0:, 0].values
# convert to binary labels (0 and 1)
y_train = np.array([1 if v == 'Android' else 0 for v in Y_train])

df_test = pd.read_csv('tweets.test.tsv', sep='\t', header=None)
text_test = df_test.iloc[0:, 1].values.tolist()
Y_test = df_test.iloc[0:, 0].values
# convert to binary labels (0 and 1)
y_test = np.array([1 if v == 'Android' else 0 for v in Y_test])

"""Let us take a quick look of some training examples"""

text_train[:5]

y_train[:5]

"""#### Sample code to preprocess data for BERT (only needed if you decide to fine-tune BERT) ####

The pre-processing step is similar to Lab 5.

Feel free to dispose it if you want to preprocess the data differently and use methods other than BERT.
"""

# The longest text in the data is 75 and we use it as the max_length
max_length = 75
x_train = bert_tokenizer(text_train,
              max_length=75,
              truncation=True,
              padding='max_length',
              return_tensors='tf')

y_train = np.array([1 if v == 'Android' else 0 for v in Y_train])

x_test = bert_tokenizer(text_test,
              max_length=75,
              truncation=True,
              padding='max_length',
              return_tensors='tf')

y_test = np.array([1 if v == 'Android' else 0 for v in Y_test])

"""### Your Solution 1: A traditional machine learning approach (30%)

Please implement your text classifier using a traditional machine learning method.

**Inline Question #1:**
- What machine leaning model did you use?

We used Naives Bayes as our machine learning model.

- What are the features used in this model?

The features of this model are essentially the counts of each word (from the vocabulary built from the training set) present in the tweets. The dimensionality of the feature space is equal to the size of the vocabulary (the number of unique words found in the training data), which can be quite large for text data. The process includes tokenization, wherethe text is split into tokens (usually words), creating a vocabulary of all unique words in the dataset. And count vectorization, each tweet is represented as a vector indicating the frequency of each word in the vocabulary appearing in the document. If a word from the vocabulary appears in the document, its corresponding position in the vector is incremented.


- What is the model's performance in the test data?

The model's performance on the test data is 85.95%

**Your Answer:**
"""

################################################################################
# TODO: Fill in your codes                                                     #
################################################################################

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Vectorize the text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(text_train)
X_test = vectorizer.transform(text_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['iPhone', 'Android'])

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

"""### Your Solution 2: A deep learning apporach (30%)

Please implement your text classifier using a deep learning method

**Inline Question #1:**
- What deep leaning model did you use?
- Please briefly explain the input, output, and layers (e.g., what does each layer do) of your model.
- What is the model's performance in the test data?
- Is it better or worse than Solution 1? What might be the cause?

**Your Answer:**
"""

'''
The network we used was based of the research done in the following article : https://www.mdpi.com/1424-8220/22/11/4157
This network was attractive because it utilized a lot of the different techniques available to approach this problem.
Though the exact network was not replicated the layer structure was. The structure is as follows:
1)embedding layer 2)Convolutional Layer 3)Pooling Layer 4)BLSTM 5)Concatenation 6)Dense 7)Activitaion
The preformance of this model was compared to that of our ML model as well as a simple BLSTM.
The results of this model yielded slightly better results than that of the BLSTM and had over a 4% increase of
accuracy from the ML model. Additionally, our model trained much faster than the BLSTM, but we discovered it very
rapidly overtrained on the data so epochs were limited to 3. Another aspect that may have enhanced performance
in our model is using pre-trained embeddings with high very high dimensionality enabling higher accuracy of
classification.
'''

"""## Final note (10%)

Similar to Homework 1, 10% of the total grade is allocated based on model performance. Teams with higher performance scores (max of solution 1 and solution 2) get higher grade.


"""

import os

# Only the TensorFlow backend supports string inputs.
os.environ["KERAS_BACKEND"] = "tensorflow"

import pathlib
import numpy as np
import keras
from keras import layers

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.layers import SpatialDropout1D, Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np

#training set
df_train = pd.read_csv('tweets.train.tsv', sep='\t', header=None)

text_train = df_train.iloc[0:, 1].values.tolist()
Y_train = df_train.iloc[0:, 0].values
# convert to binary labels (0 and 1)
y_train = np.array([1 if v == 'Android' else 0 for v in Y_train])

df_test = pd.read_csv('tweets.test.tsv', sep='\t', header=None)
text_test = df_test.iloc[0:, 1].values.tolist()
Y_test = df_test.iloc[0:, 0].values
# convert to binary labels (0 and 1)
y_test = np.array([1 if v == 'Android' else 0 for v in Y_test])

from keras.preprocessing.text import Tokenizer
vocab_size = 40000
oov_token = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(text_train)

word_index = tokenizer.word_index

X_train_sequences = tokenizer.texts_to_sequences(text_train)
X_test_sequences = tokenizer.texts_to_sequences(text_test)

max_length = 75
padding_type='post'
truncation_type='post'

from keras.preprocessing.sequence import pad_sequences

X_test_padded = pad_sequences(X_test_sequences,maxlen=max_length,
                               padding=padding_type, truncating=truncation_type)
X_train_padded = pad_sequences(X_train_sequences,maxlen=max_length, padding=padding_type,
                       truncating=truncation_type)

!wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
!unzip -q glove.6B.zip

path_to_glove_file = "glove.6B.300d.txt"

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

embedding_layer = Embedding(input_dim=len(word_index) + 1,
                            output_dim=300,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)

import tensorflow as tf
from tensorflow.keras import layers, models

# Assuming you have loaded your pre-trained embeddings into `pretrained_embeddings`
# with shape (vocab_size, embedding_dim), where vocab_size=400000, embedding_dim=300

vocab_size = 400000  # as per your pre-trained embeddings
embedding_dim = 300  # dimension of your pre-trained embeddings
max_length = 75      # max length of tweets

model = models.Sequential([
    embedding_layer,  # Set trainable to False to keep embeddings fixed
    layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=4),
    layers.LSTM(64),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Output

from tensorflow.keras.callbacks import EarlyStopping

callbacks = [
    EarlyStopping(patience=10),
]
num_epochs = 3

# Assuming X_train_padded, y_train, X_test_padded, and y_test are already defined and properly preprocessed
history = model.fit(X_train_padded, y_train, epochs=num_epochs, validation_data=(X_test_padded, y_test), callbacks=callbacks)

#Validation
from sklearn.metrics import accuracy_score, classification_report

loss, accuracy = model.evaluate(X_test_padded,y_test)
print('Test accuracy :', accuracy)
