# -*- coding: utf-8 -*-
"""hw2-Nathan-Ryan.ipynb


# MSBC5180 HW2 Coding Part: Linear Classification

### Solution by: Nathan Ryan (ft. Caroline Jones)

## Assignment overview

In this assignment, you will build a classifier that tries to infer whether tweets from [@realDonaldTrump](https://twitter.com/realDonaldTrump) were written by Trump himself or by a staff person.
This is an example of binary classification on a text dataset.

It is known that Donald Trump uses an Android phone, and it has been observed that some of his tweets come from Android while others come from other devices (most commonly iPhone). It is widely believed that Android tweets are written by Trump himself, while iPhone tweets are written by other staff. For more information, you can read this [blog post by David Robinson](http://varianceexplained.org/r/trump-tweets/), written prior to the 2016 election, which finds a number of differences in the style and timing of tweets published under these two devices. (Some tweets are written from other devices, but for simplicity the dataset for this assignment is restricted to these two.)

This is a classification task known as "authorship attribution", which is the task of inferring the author of a document when the authorship is unknown. We will see how accurately this can be done with linear classifiers using word features.

## Getting started

In this assignment, you will experiment with perceptron and logistic regression in `sklearn`. Much of the code has already been written for you. We will use a class called `SGDClassifier` (which you should read about in the [sklearn documentation](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)), which  implements stochastic gradient descent (SGD) for a variety of loss functions, including both perceptron and logistic regression, so this will be a way to easily move between the two classifiers.

The code below will load the datasets. There are two data collections: the "training" data, which contains the tweets that you will use for training the classifiers, and the "testing" data, which are tweets that you will use to measure the classifier accuracy. The test tweets are instances the classifier has never seen before, so they are a good way to see how the classifier will behave on data it hasn't seen before. However, we still know the labels of the test tweets, so we can measure the accuracy.

For this problem, we will use what are called "bag of words" features, which are commonly used when doing classification with text. Each feature is a word, and the value of a feature for a particular tweet is number of times the word appears in the tweet (with value $0$ if the word does not appear in the tweet).

Run the block of code below to load the data. You don't need to do anything yet. Move on to "Problem 1" next.
"""

from google.colab import drive
drive.mount('/content/drive/')

import os
os.chdir('/content/drive/MyDrive/WorkingDirectory/')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#training set
df_train = pd.read_csv('tweets.train.tsv', sep='\t', header=None)

Y_train = df_train.iloc[0:, 0].values
text_train = df_train.iloc[0:, 1].values

vec = CountVectorizer()
X_train = vec.fit_transform(text_train)
feature_names = np.asarray(vec.get_feature_names_out())

#testing set
df_test = pd.read_csv('tweets.test.tsv', sep='\t', header=None)
Y_test = df_test.iloc[0:, 0].values
text_test = df_test.iloc[0:, 1].values

X_test = vec.transform(text_test)

df_train.head()

"""## Problem 1: Understand the data

Before doing anything else, take time to understand the code above.

The variables `df_train` and `df_test` are dataframes that store the training (and testing) datasets, which are contained in tab-separated files where the first column is the label and the second column is the text of the tweet.

The [`CountVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) class converts the raw text into a bag-of-words into a feature vector representation that `sklearn` can use.

You should print out the values of the variables and write any other code needed to answer the following questions.

#### Deliverable 1.1: How many training instances are in the dataset? How many test instances?


"""

print(f'training instances :  {df_train.shape[0]}')
print(f'test instances :  {df_test.shape[0]}')

"""#### Deliverable 1.2: How many features are in the training data?




"""

len(feature_names)

"""#### Deliverable 1.3: What is the distribution of labels in the training data? That is, what percentage of instances are 'Android' versus 'iPhone'?


"""

df_train.describe() #1339 android tweets
print(f'percent android : {round(1339/2593 * 100, 2)}%')
print(f'percent iphone : {round((2593-1339)/2593*100, 2)}%')

"""## Problem 2: Perceptron

The code below trains an [`SGDClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) using the perceptron loss, then it measures the accuracy of the classifier on the test data, using `sklearn`'s [`accuracy_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) function.

The `fit` function trains the classifier. The feature weights are stored in the `coef_` variable after training. The `predict` function of the trained `SGDClassifier` outputs the predicted label for a given instance or list of instances.

Additionally, this code displays the features and their weights in sorted order, which you may want to examine to understand what the classifier is learning. In this dataset, the $\textrm{Android}$ class is considered the "negative" class because it comes first in the data.

There are 3 keyword arguments that have been added to the code below. It is important you keep the same values of these arguments whenever you create an `SGDClassifier` instance in this assignment so that you get consistent results. They are:

- `max_iter` is one of the stopping criteria, which is the maximum number of iterations/epochs the algorithm will run for.

- `tol` is the other stopping criterion, which is how small the difference between the current loss and previous loss should be before stopping.

- `random_state` is a seed for pseudorandom number generation. The algorithm uses randomness in the way the training data are sorted, which will affect the solution that is learned, and even the accuracy of that solution.

Wait a minute $-$ in class we learned that the loss function is convex, so the algorithm will find the same minimum regardless of how it is trained. Why is there random variation in the output? The reason is that even though there is only one minimum value of the loss, there may be different weights that result in the same loss, so randomness is a matter of tie-breaking. What's more, while different weights may have the same loss, they could lead to different classification accuracies, because the loss function is not the same as accuracy. (Unless accuracy was your loss function... which is possible, but uncommon because it turns out to be a difficult function to optimize.)

Note that different computers may still give different answers, despite keeping these settings the same, because of how pseudorandom numbers are generated with different operating systems and Python environments.

To begin, run the code in the cell below without modification.


<br />


"""

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score

classifier = SGDClassifier(loss='perceptron', max_iter=1000, tol=1.0e-12, random_state=123, eta0=100)
classifier.fit(X_train, Y_train)

print("Number of SGD iterations: %d" % classifier.n_iter_)
print("Training accuracy: %0.6f" % accuracy_score(Y_train, classifier.predict(X_train)))
print("Testing accuracy: %0.6f" % accuracy_score(Y_test, classifier.predict(X_test)))

'''
print("\nFeature weights:")
args = np.argsort(classifier.coef_[0])
for a in args:
    print(" %s: %0.4f" % (feature_names[a], classifier.coef_[0][a]))
'''

"""#### Deliverable 2.1: Based on the training accuracy, do you conclude that the data are linearly separable? Why or why not?"""

#It is difficult to just conclude from the training data that the data is linearly separable. It may be that the model chosen -
# is overfitting to the training data. However, the high Testing Accuracy suggests that the model is effective and furthermore linearly separable

"""#### Deliverable 2.2: Which feature most increases the likelihood that the class is 'Android' and which feature most increases the likelihood that the class is 'iPhone'?"""

print("\nFeature weights:")
args = np.argsort(classifier.coef_[0])
for a in args:
    print(" %s: %0.4f" % (feature_names[a], classifier.coef_[0][a]))

"""One technique for improving the resulting model with perceptron (or stochastic gradient descent learning in general) is to take an average of the weight vectors learned at different iterations of the algorithm, rather than only using the final weights that minimize the loss. That is, calculate $\bar{\mathbf{w}} =\frac{ \sum_{t=1}^T \mathbf{w}^{(t)}}{T}$ where $\mathbf{w}^{(t)}$ is the weight vector at iteration $t$ of the algorithm and $T$ is the number of iterations, and then use $\bar{\mathbf{w}}$ when making classifications on new data.

To use this technique in your classifier, add the keyword argument `average=True` to the `SGDClassifier` function. Try it now.

#### Deliverable 2.3: Compare the initial training/test accuracies to the training/test accuracies after doing averaging. What happens? Why do you think averaging the weights from different iterations has this effect?

[your answer here]


"""

classifier = SGDClassifier(loss='perceptron', max_iter=1000, tol=1.0e-12, random_state=123, eta0=100, average = True)
classifier.fit(X_train, Y_train)

print("Number of SGD iterations: %d" % classifier.n_iter_)
print("Training accuracy: %0.6f" % accuracy_score(Y_train, classifier.predict(X_train)))
print("Testing accuracy: %0.6f" % accuracy_score(Y_test, classifier.predict(X_test)))

#After averaging, there was a marginal decrease in training accuracy but over .05% increase in testing accuracy

# I believe that testing accuracy was improved and training accuracy was decreased because averaging the weights instead of using the final weight -
# decreased bias in the model preventing overfitting on the training set and improving preformance on the test set

"""## Problem 3: Logistic regression

For this problem, create a new `SGDClassifier`, this time setting the `loss` argument to `'log_loss'`, which will train a logistic regression classifier. Set `average=False` for the remaining problems.

Once you have trained the classifier, you can use the `predict` function to get the classifications, as with perceptron. Additionally, logistic regression provides probabilities for the predictions. You can get the probabilities by calling the `predict_proba` function. This will give a list of two numbers; the first is the probability that the class is $\textrm{Android}$ and the second is the probability that the class is $\textrm{iPhone}$.


For the first task, add the keyword argument `alpha` to the `SGDClassifier` function. This is the regularization strength, called $\lambda$ in lecture. If you don't specify `alpha`, it defaults to $0.0001$. Experiment with other values and see how this affects the outcome.

#### Deliverable 3.1: Calculate the training and testing accuracy when `alpha` is one of $[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]$. Create a plot where the x-axis is `alpha` and the y-axis is accuracy, with two lines (one for training and one for testing). You can borrow the code from HW1 for generating plots in Python. Use [a log scale for the x-axis](https://matplotlib.org/examples/pylab_examples/log_demo.html) so that the `alpha` values are spaced evenly. `plt.semilogx()` is the function you should use if you import `matplotlib.pyplot` as `plt`.




"""

al = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0]
train_accuracy = []
test_accuracy = []

for a in al :
  classifier = SGDClassifier(loss = 'log_loss', max_iter = 1000, tol = 1.0e-12, random_state = 123, eta0 = 100, alpha = a)
  classifier.fit(X_train, Y_train)

  train_acc = accuracy_score(Y_train, classifier.predict(X_train))
  test_acc = accuracy_score(Y_test, classifier.predict(X_test))

  train_accuracy.append(train_acc)
  test_accuracy.append(test_acc)

  print(f"alpha = {a}")
  print("Training accuracy: %0.6f" % accuracy_score(Y_train, classifier.predict(X_train)))
  print("Testing accuracy: %0.6f" % accuracy_score(Y_test, classifier.predict(X_test)))

import matplotlib.pyplot as plt

plt.semilogx(al, train_accuracy, label = 'Training Accuracy', color = 'blue')
plt.semilogx(al, test_accuracy, label = 'Test Accuracy', color = 'red')
plt.xlabel('Alpha (log scale)')
plt.ylabel('Accuracy')

plt.legend()
plt.show()

"""#### Deliverable 3.2: Examine the classifier probabilities using the `predict_proba` function when training with different values of `alpha`. You can examine the first test instance for this. What do you observe? How does `alpha` affect the prediction probabilities, and why do you think this happens?

"""

al = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0]

for a in al :
  classifier = SGDClassifier(loss = 'log_loss', max_iter = 1000, tol = 1.0e-12, random_state = 123, eta0 = 100, alpha = a)
  classifier.fit(X_train, Y_train)

  print(f'Classifier Probabilities for alpha = {a} :')
  print(classifier.predict_proba(X_test)[0])

# As alpha grows larger, we observe the Classifier Probabilities (of the 1st instance) converge to and almost 50/50 odds that the tweet is
# either iPhone or Android.
# We can conclude this happens as alpha grows large training/testing accuracy dramatically decreases. As this is a binary classifier the probability
# that the classifier will conclude iPhone or Android will converge to 50% probability (a random guess!)

"""Now remove the `alpha` argument so that it goes back to the default value. We'll now look at the effect of the learning rate. By default, `sklearn` uses an "optimal" learning rate based on some heuristics that work well for many problems. However, it can be good to see how the learning rate can affect the algorithm.

For this task, add the keyword argument `learning_rate` to the `SGDClassifier` function and set the value to `invscaling`. This defines the learning rate at iteration $t$ as: $\eta_t = \frac{\eta_0}{t^a}$, where $\eta_0$ and $a$ are both arguments you have to define in the `SGDClassifier` function, called `eta0` and `power_t`, respectively. Experiment with different values of `eta0` and `power_t` and see how they affect the number of iterations it takes the algorithm to converge. You will often find that it will not finish within the maximum of $1000$ iterations.
"""

al = np.random.randint(70000, 110000, size = 10)
als = np.sort(al)

for a in als :
  classifier = SGDClassifier(loss='perceptron', max_iter=1000, tol=1.0e-12, random_state=123, eta0=  a, power_t = 3, learning_rate = 'invscaling')
  classifier.fit(X_train, Y_train)

  print(f'eta0 = {a}')
  print("Number of SGD iterations: %d" % classifier.n_iter_)
  print("Training accuracy: %0.6f" % accuracy_score(Y_train, classifier.predict(X_train)))
  print("Testing accuracy: %0.6f" % accuracy_score(Y_test, classifier.predict(X_test)))

al = [10,100,1000,10000]
bl = [0.5,1,2]


for a in al :
  for b in bl :
    classifier = SGDClassifier(loss='perceptron', max_iter=1000, tol=1.0e-12, random_state=123, eta0=  a, power_t = b, learning_rate = 'invscaling')
    classifier.fit(X_train, Y_train)

    print(f'eta0 = {a}')
    print(f'power_t = {b}')
    print("Number of SGD iterations: %d" % classifier.n_iter_)
    print("Training accuracy: %0.6f" % accuracy_score(Y_train, classifier.predict(X_train)))
    print("Testing accuracy: %0.6f" % accuracy_score(Y_test, classifier.predict(X_test)))

"""#### Deliverable 3.3: Fill in the table below with the number of iterations for values of `eta0` in $[10.0, 100.0, 1000.0, 10000.0]$ and values of `power_t` in $[0.5, 1.0, 2.0]$. You may find it easier to write python code that can output the markdown for the table, but if you do that place the output here. If it does not converge within the maximum number of iterations (set to $1000$ by `max_iter`), record $1000$ as the number of iterations.

| `eta0`   | `power_t` | # Iterations |
|-----------|-----------|--------------|
| $10.0$    | $0.5$     |        25    |
| $10.0$    | $1.0$     |         1000     |
| $10.0$    | $2.0$     |        1000      |
| $100.0$   | $0.5$     |        38      |
| $100.0$   | $1.0$     |        1000      |
| $100.0$   | $2.0$     |        1000      |
| $1000.0$  | $0.5$     |         35     |
| $1000.0$  | $1.0$     |         1000     |
| $1000.0$  | $2.0$     |         1000     |
| $10000.0$ | $0.5$     |         49     |
| $10000.0$ | $1.0$     |         33     |
| $10000.0$ | $2.0$     |         1000     |



<br/>

#### Deliverable 3.4: Describe how `eta0` and `power_t` affect the learning rate based on the formula (e.g., if you increase `power_t`, what will this do to the learning rate?), and connect this to what you observe in the table above.
"""

# The learning rate at iteration  𝑡  is:  𝜂_𝑡=𝜂_0/𝑡^𝑎
# In our classifier model: eta0 = 𝜂_0 & power_t = log_t(t^a) = a (the log of base t of t to the power of a equals a)
# Thus when n_0 is large and a is small : 𝜂_𝑡 is large
## When 𝜂_𝑡 is large the number of iterations decreases and accuracy improves
### however if 𝜂_0 is small and t^a small there is a too high a bias on the training set and lower test accuracy
### When 𝜂_𝑡 is large and t^a the model reduces overfitting and training accuracy decreases while test accuracy increases

"""Now remove the `learning_rate`, `eta0`, and `power_t` arguments so that the learning rate returns to the default setting. For this final task, we will experiment with how high the probabiity must be before an instance is classified as positive.

The code below includes a function called `threshold` which takes as input the classification probabilities of the data (called `probs`, which is given by the function `predict_proba`) and a threshold (called `tau`, a scalar that should be a value between $0$ and $1$). It will classify each instance as $\textrm{Android}$ if the probability of being $\textrm{Android}$ is greater than `tau`, otherwise it will classify the instance as $\textrm{iPhone}$. Note that if you set `tau` to $0.5$, the `threshold` function should give you exactly the same output as the classifier `predict` function.

You should find that increasing the threshold causes the accuracy to drop. This makes sense, because you are classifying some things as $\textrm{iPhone}$ even though it's more probable that they are $\textrm{Android}$. So why do this? Suppose you care more about accurately identifying the $\textrm{Android}$ tweets and you don't care as much about `iPhone` tweets. You want to be confident that when you classify a tweet as $\textrm{Android}$ that it really is $\textrm{Android}$.

There is a metric called _precision_ which measures something like accuracy but for one specific class. Whereas accuracy is the percentage of tweets that were correctly classified, the precision of $\textrm{Android}$ would be the percentage of tweets classified as $\textrm{Android}$ that were correctly classified. (In other words, the number of tweets classified as $\textrm{Android}$ whose correct label was $\textrm{Android}$, divided by the number of tweets classified as $\textrm{Android}$.)

You can use the [`precision_score`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score) function from `sklearn` to calculate the precision. It works just like the `accuracy_score` function, except you have to add an additional keyword argument, `pos_label='Android'`, which tells it that $\textrm{Android}$ is the class you want to calculate the precision of.
#### Deliverable 3.5: Calculate the testing precision when the value of `tau` for thresholding is one of $[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]$. Create a plot where the x-axis is `tau` and the y-axis is precision.



"""

def threshold(probs, tau):
    return np.where(probs[:,0] > tau, 'Android', 'iPhone')

classifier = SGDClassifier(loss = 'log_loss', max_iter = 1000, tol = 1.0e-12, random_state = 123)
classifier.fit(X_train, Y_train)

probs = classifier.predict_proba(X_test)

tau = [0.5,0.6,0.7,0.8,0.9,0.95,0.99]

for t in tau :
  Threshold = threshold(probs, t)

  print(f"tau = {t}")
  print("Testing precision: %0.6f" % precision_score(Y_test, Threshold, pos_label = 'Android'))

precision = []

for t in tau :
  Threshold = threshold(probs, t)

  precision_sc = precision_score(Y_test, Threshold, pos_label = 'Android')
  precision.append(precision_sc)

plt.plot(tau, precision, label = 'Testing Precision'  )
plt.xlabel('tau')
plt.ylabel('Precision')

plt.show()

"""
#### Deliverable 3.6: Describe what you observe with thresholding (e.g., what happens to precision as the threshold increases?), and explain why you think this happens.
"""

'''
 The precision converges to 100%. As the tau converges to 1 only values of 'Android' will be classified as the threshold will discriminate all -
 'iPhone' values such that there are only 'Android' values. Interestingly there is a dip at tau = .95 which may merit investigation in the linear -
 seperation of the data set by the model
 '''

"""## Problem 4: Sparse learning

Add the `penalty` argument to `SGDClassifier` and set the value to `'l1'`, which tells the algorithm to use L1 regularization instead of the default L2. Recall from lecture that L1 regularization encourages weights to stay at exactly $0$, resulting in a more "sparse" model than L2. You should see this effect if you examine the values of `classifier.coef_`. We are still doing logistic regression here.

#### Deliverable 4.1: Write a function to calculate the number of features whose weights are nonzero when using L1 regularization. Calculate the number of nonzero feature weights when `alpha` is one of $[0.00001, 0.0001, 0.001, 0.01, 0.1]$. Create a plot where the x-axis is `alpha` and the y-axis is the number of nonzero weights, using a log scale for the x-axis.


"""

classifier = SGDClassifier(loss = 'log_loss', max_iter = 1000, tol = 1.0e-12, random_state = 123, penalty = 'l1')
classifier.fit(X_train, Y_train)
classifier.coef_ # yep a bunch of 0's in the coefs

alpha = [0.00001,0.0001,0.001,0.01,0.1]
num_coeffs = []

for a in alpha :
  classifier = SGDClassifier(loss = 'log_loss', max_iter = 1000, tol = 1.0e-12, random_state = 123, penalty = 'l1', alpha = a)
  classifier.fit(X_train, Y_train)

  non_zero = np.nonzero(classifier.coef_[0])[0]

  #print(non_zero)
  coeffs = len(non_zero)
  num_coeffs.append(coeffs)

  #for coef in non_zero:
   #     print("%s: %0.4f" % (feature_names[coef], classifier.coef_[0][coef]))
# I now have an array of features with non-zero coeffs
# for my graph I just need the amounts in accordance with each alpha

plt.semilogx(alpha, num_coeffs, label = 'Num Non-Zero Coeffs')
plt.xlabel('alpha')
plt.ylabel('Non-Zero Coeffs')

plt.show()
