# -*- coding: utf-8 -*-
"""hw5-nathan-ryan.ipynb

# MSBC5180: Ensemble Learning

In this assignment, you will continue working with the Twitter sentiment dataset from HW4. This time, you will build a classifier that combines the individual classifiers submitted by everyone in the class.

## Combined Dataset


The probabilities from some of the submissions from HW4 have been put together for this assignment. The format is a CSV file where the first column is the label, and subsequent columns are classifier probabilities. Each three-column sequence is the probability of negative ($-1$), neutral ($0$), and positive ($1$), in that order. For example, column 2 (where column 1 is the label) is the negative probability, column 3 is the neural probability, and cololum column 4 is the positive probability from the first submission. Column 5 is the negative probability of the second submission, column 6 is the neutral probability of the second submission, and so on. There are two files: the first should be used for training and cross-validation, and the second should be used for testing.

As usual, run the code below to load the data. The accuracies of each individual system are also calculated.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('tweet_predictions_cv.csv', header=None)
df_test = pd.read_csv('tweet_predictions_test.csv', header=None)

Y_train = df_train.iloc[0:, 0].values
X_train = df_train.iloc[0:, 1:].values

Y_test = df_test.iloc[0:, 0].values
X_test = df_test.iloc[0:, 1:].values

for i in np.arange(0, len(X_train[0]), 3):
    print("Submission %d:" % (1 + int(i/3)))
    predictions_cv = [np.argmax(x)-1 for x in X_train[0:, i:i+3]]
    print(" Validation accuracy: %0.6f" % accuracy_score(Y_train, predictions_cv))
    predictions_test = [np.argmax(x)-1 for x in X_test[0:, i:i+3]]
    print(" Test accuracy: %0.6f" % accuracy_score(Y_test, predictions_test))

"""## Problem 1: Ensemble Classifier: Stacking

First, build a classifier that uses the probabilities from the 36 submissions as features. Since each submission contains 3 probabilities, there are 108 total features.

Following HW4, you should use multinomial logistic regression as the classifier. Use `sklearn`'s [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) class, setting the `multi_class` argument to `'multinomial'`, the `solver` argument to `'lbfgs'`, and the `random_state` argument to `123` (as usual).

Additionally, use [`GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to select the `C` parameter using 5-fold cross-validation. For the grid search, try the following values for `C`: ${0.1, 0.2, 0.3, 0.4, \ldots, 1.8, 1.9, 2.0}$. (You can easily generate this list of values using [`numpy.arange`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.arange.html).) When making predictions on the test data, you should use the optimal classifier tuned during cross-validation.

You may wish to refer to the HW4 code to get started, since the code will be similar.

#### Deliverable 1.1: Implement the ensemble classifier as described, and calculate both the cross-validation accuracy and test accuracy.

[output below]

#### Deliverable 1.2: Examine the validation and test accuracies of the individual submissions above. How do these accuracies compare to the validation and test accuracy of your ensemble classifier?


#### Deliverable 1.3: Based on what was discussed in lecture, explain these results. If the ensemble outperformed the individual classifiers, explain why ensembles are able to do this. If the ensemble did not outperform the individual classifiers, explain why this particular ensemble might not have been effective.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#1.1
params = [{'C' : np.arange(0.1,2.1,0.1)}]
lgr = LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter = 500,random_state=123)

gs_classifier = GridSearchCV(lgr,params,cv=5)
gs_classifier.fit(X_train,Y_train)

print("Validation accuracy: %0.6f" % gs_classifier.best_score_)
print("Test accuracy: %0.6f" % accuracy_score(Y_test, gs_classifier.predict(X_test)))

#1.2
'''
The ensemble classifier greatly outpreforms the individual classifiers in both validation and test accuracy.
The highest score achieved for both of the individual classifiers was 72% and 75% respectively vs.
a score of 81% and 80% for the ensemble classifier
'''

#1.3
'''
Ensembles may outpreform individual classifiers for a multitude of reasons. One reason the ensemble outpreformed
the individual classifiers is due to the diversity of the individual classifiers. Each classifier had a range of
accuracy scores (both validation and test) some ranging ~68% to ~79%. When there is high diversity in an ensemble
classifier it is able to utilize the strengths or reduce weakness of the resultant classifier or prevent
overfitting on the dataset
'''

"""## Problem 2: Dimensionality Reduction

Since the features are continuous-valued and correlated with each other, this feature set is a good candidate for dimensionality reduction with principal component analysis (PCA). You will experiment with PCA here.

Use the [`sklearn.decomposition.PCA`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to transform the feature vectors (`X_train` and `X_test`) using PCA.  You should fit PCA with the training data, and then transform the feature vectors of both the training and test data. This will require a combination of the `fit`, `transform`, and/or `fit_transform` functions.

When creating a `PCA` object, you set the number of components (that is, the dimensionality of the feature vectors) with the `n_components` argument. Additionally, set `random_state` to `123`.

You should run the same classifier from Problem 1 on the PCA-reduced data. You should continue to use `GridSearchCV` to tune `C`.

#### Deliverable 2.1: Apply PCA to the data and calculate the validation and test accuracies when the number of components is each of: $1, 2, 10, 20, 30, 40, 50, 100$.

[you may wish to plot these results, but it is not required as long as your results are readable]
"""

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

components = [1,2,10,20,30,40,50,100]
for p in components:
  pipe_lr = make_pipeline(PCA(n_components=p,random_state=123),lgr)
  pipe_lr.fit(X_train,Y_train)

  print(f'components = {p}')
  print("Validation accuracy: %0.6f" % accuracy_score(Y_train, pipe_lr.predict(X_train)))
  print("Test accuracy: %0.6f" % accuracy_score(Y_test, pipe_lr.predict(X_test)))
  print('')
