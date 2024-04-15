# -*- coding: utf-8 -*-
"""Lab9_CV_and_Grid_Search.ipynb

# Learning Best Practices for Model Evaluation and Hyperparameter Tuning
"""

# Commented out IPython magic to ensure Python compatibility.
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

"""# Streamlining workflows with pipelines

## Loading the Breast Cancer Wisconsin dataset

Data Set Information:

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. A few of the images can be found at [Web Link]

Separating plane described above was obtained using Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree Construction Via Linear Programming." Proceedings of the 4th Midwest Artificial Intelligence and Cognitive Science Society, pp. 97-101, 1992], a classification method which uses linear programming to construct a decision tree. Relevant features were selected using an exhaustive search in the space of 1-4 features and 1-3 separating planes.

The actual linear program used to obtain the separating plane in the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

This database is also available through the UW CS ftp server:
ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/


Attribute Information:

1) ID number

2) Diagnosis (M = malignant, B = benign)

3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)

b) texture (standard deviation of gray-scale values)

c) perimeter

d) area

e) smoothness (local variation in radius lengths)

f) compactness (perimeter^2 / area - 1.0)

g) concavity (severity of concave portions of the contour)

h) concave points (number of concave portions of the contour)

i) symmetry

j) fractal dimension ("coastline approximation" - 1)
"""

import pandas as pd

# df = pd.read_csv('https://archive.ics.uci.edu/ml/'
 #                'machine-learning-databases'
  #                '/breast-cancer-wisconsin/wdbc.data', header=None)

# if the Breast Cancer dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

df = pd.read_csv('wdbc.data', header=None)
df.head()

"""Create label:
1. Import `LabelEncoder` from `sklear.preprocessing`
2. Create an instance of LabelEncoder called `le`
3. Take the second column (index 1) as the lable `y`
4. Apply le on y to covert categorial data
5. Take the data from the third column to the 32nd column as the feature matrix `X`
"""

from sklearn.preprocessing import LabelEncoder

y = df.loc[:,1].values
X = df.loc[:,2:].values

le = LabelEncoder()
y = le.fit_transform(y)

le.classes_

le.transform(['M','B'])

"""Train and test set splitting with 20% data in the test set
1. Use train_test_split from sklearn.model_selection to split the data into training and test sets
2. 20% data in the test set
3. Also, we need `stratify=y, random_state =1`
"""

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, stratify = y, random_state = 1)

"""## Combining transformers and estimators in a pipeline"""

Image(filename='pipeline.png', width=500)

"""Use make_pipeline from scikit-learn to create a work pipeline, which can have multiple functions in a process."""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

"""We create a pipeline for creating a logistic regression model called `pipe_lr`
1. It starts with StandardScaler(), then conducts PCA with two principle components. After that, it use LogisticRegression() with `random_state =1`
2. We can use fit() method to train a model. It automatically handles the preprocessing steps.
3. Its predict() method can be directly applied to make prediction on a new data set.
"""

pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=1))

pipe_lr.fit(X_train,y_train)

y_pred = pipe_lr.predict(X_test)

pipe_lr.score(X_test,y_test)

"""# Using k-fold cross validation to assess model performance

## K-fold cross-validation
"""

Image(filename='k-fold_CV.png', width=500)

"""1. We use `StratifiedKFold` (`KFold` can be used for regression problems) from sklearm.model_selection to create the splits (folds) with the split() method.
2. We can utilize the pipe line we created earlier to loop through each split
3. We will get the model's performance on each split and the overall performance
"""

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True).split(X_train,y_train)

scores = []

for k, (train,validation_fold) in enumerate(kfold):
  pipe_lr.fit(X_train[train],y_train[train])
  score = pipe_lr.score(X_train[validation_fold],y_train[validation_fold])
  scores.append(score)

print(np.mean(score))

print(np.std(scores))

"""We can use cross_val_score to do this"""

import numpy as np
from sklearn.model_selection import cross_val_score

scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

"""<br>
<br>

# Debugging algorithms with learning curves

<br>
<br>

## Diagnosing bias and variance problems with learning curves
"""

Image(filename='bV.png', width=600)

"""Get the info for learning curve
1. We can use `learning_curve` from `sklearn.model_selection` for this.
"""

from sklearn.model_selection import learning_curve

pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l2', random_state = 1))
train_sizes, train_scores, test_scores = learning_curve(estimator = pipe_lr, X = X_train, y = y_train,
                                                        train_sizes = np.linspace(0.1,1,10),
                                                        cv = 10, n_jobs = 1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
#plt.savefig('images/06_05.png', dpi=300)
plt.show()

"""<br>
<br>

## Addressing over- and underfitting with validation curves

Get the info for validation curve
1. We can use `validation_curve` from `sklearn.model_selection` for this.
2. Tune parameter `losigticregression__C` with `param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`
"""

from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1, 10, 100]
train_scores, test_scores = validation_curve(estimator = pipe_lr, X = X_train, y = y_train,
                                             param_name = 'logisticregression__C',
                                             param_range = param_range,
                                             cv = 10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.tight_layout()
# plt.savefig('images/06_06.png', dpi=300)
plt.show()

"""<br>
<br>

# Fine-tuning machine learning models via grid search

<br>
<br>

## Tuning hyperparameters via grid search

We use `GridSearchCV` from sklearn.model_selection for this. We will look at one example with Support Vector Classifier (SVC from sklearm.svm)
"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state = 1))

param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid_comb = [{'svc__C':param_range,
                    'svc__kernel' : ['linear']},
                   {'svc__C': param_range,
                    'svc__gamma' : param_range,
                    'svc__kernel' : ['rbf']}]

gs = GridSearchCV(estimator = pipe_svc,
                  param_grid = param_grid_comb,
                  scoring = 'accuracy',
                  cv = 10, n_jobs = -1)
gs = gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)

"""Use the best one for testing! the best_estimator_ will already be refit to the complete training set because of the refit=True setting in GridSearchCV (refit=True by default)."""

clf = gs.best_estimator_

clf.score(X_test, y_test)

"""<br>
<br>

## Algorithm selection with nested cross-validation
"""

Image(filename='nested_CV.png', width=500)

"""We will use a 5 X 2 nested cross validation. Thus, the outer loop has 5 iterations and the inner loop has 2 iteration.

We will compare the performance of SVC and decision tree to pick a better model.
"""

gs = GridSearchCV(estimator = pipe_svc, param_grid = param_grid_comb, scoring = 'accuracy', cv = 2)
scores = cross_val_score(gs, X_train, y_train, scoring = 'accuracy', cv = 5)

scores

np.mean(scores)

np.std(scores)

"""Please try to apply nested cross-validation on decison tree model following the SVC example above:
1. the decision tree has `random_state=0`
2. we will tune the hyperparameter `max_depth` with values [1,2,3,4,5,6,7,None]
3. scoring is `accuracy`
"""

from sklearn.tree import DecisionTreeClassifier

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train,
                         scoring='accuracy', cv=5)
print('Nested CV accuracy: %.3f +/- %.3f' % (np.mean(scores),
                                      np.std(scores)))

"""<br>
<br>

# Looking at different performance evaluation metrics

## Reading a confusion matrix
"""

from sklearn.metrics import confusion_matrix

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
#plt.savefig('images/06_09.png', dpi=300)
plt.show()

le.transform(['M', 'B'])

"""We conclude:

Assuming that class 1 (malignant) is the positive class in this example, our model correctly classified 71 of the samples that belong to class 0 (true negatives) and 40 samples that belong to class 1 (true positives), respectively. However, our model also incorrectly misclassified 1 sample from class 0 as class 1 (false positive), and it predicted that 2 samples are benign although it is a malignant tumor (false negatives).

## Optimizing the precision and recall of a classification model

This section shows you how to fine tune a model based on metrics other than `accuracy`
"""





"""## Plotting a receiver operating characteristic"""

# Commented out IPython magic to ensure Python compatibility.
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold #make sure you have this line
from numpy import interp

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(penalty='l2',
                                           random_state=1,
                                           C=100.0))

X_train2 = X_train[:, [4, 14]]


cv = list(StratifiedKFold(n_splits=3,
                          random_state=1, shuffle=True).split(X_train, y_train))

fig = plt.figure(figsize=(7, 5))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train],y_train[train]).predict_proba(X_train2[test])

    fpr, tpr, thresholds = roc_curve(y_train[test],
                                     probas[:, 1],
                                     pos_label=1)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,
             tpr,
             label='ROC fold %d (area = %0.2f)'
#                    % (i+1, roc_auc))

plt.plot([0, 1],
         [0, 1],
         linestyle='--',
         color=(0.6, 0.6, 0.6),
         label='random guessing')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1],
         [0, 1, 1],
         linestyle=':',
         color='black',
         label='perfect performance')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()



"""## The scoring metrics for multiclass classification"""

pre_scorer = make_scorer()
