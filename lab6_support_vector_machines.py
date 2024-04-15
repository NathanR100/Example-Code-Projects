# -*- coding: utf-8 -*-
"""Lab6_Support_Vector_Machines.ipynb"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# %matplotlib inline

# We'll define a function to draw a nice plot of an SVM
def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = svc.decision_function(xy).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3, levels=[-1, 0, 1],
           linestyles=['--', '-', '--'])
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)
    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker ='*',  s=70, linewidths= 1, facecolors='none', edgecolors='k')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)

"""# Support Vector Machines

In this lab, we'll use the ${\tt SVC}$ module from the ${\tt sklearn.svm}$ package to demonstrate the support vector classifier
and the SVM:
1. Please import SVC from sklearn.svm
"""

from sklearn.svm import SVC

"""# Support Vector Classifier

The ${\tt SVC()}$ function can be used to fit a
support vector classifier when the argument ${\tt kernel="linear"}$ is used. The ${\tt c}$ argument allows us to specify the cost of
a violation to the margin. When the ${\tt c}$ argument is **small**, then the margins
will be wide and many support vectors will be on the margin or will
violate the margin. When the ${\tt c}$ argument is **large**, then the margins will
be narrow and there will be few support vectors on the margin or violating
the margin.

We can use the ${\tt SVC()}$ function to fit the support vector classifier for a
given value of the ${\tt cost}$ parameter. Here we demonstrate the use of this
function on a two-dimensional example so that we can plot the resulting
decision boundary. Let's start by generating a set of observations, which belong
to two classes:
"""

# Generating random data: 20 observations of 2 features and divide into two classes.
np.random.seed(2)
X = np.random.randn(20,2)
y = np.repeat([1,-1], 10)

X[y == -1] = X[y == -1] +1

"""Let's plot the data to see whether the classes are linearly separable:
1. Use scatter plot to show the generated data. Show a different color for each class
"""

plt.scatter(X[:,0], X[:,1], c = y, s = 70, cmap = mpl.cm.Paired)

"""Nope; not linear. Next, we fit the support vector classifier:
1. Create a SVC instance called `svc` with parameter `C=1` and `kernel = 'linear'`
2. Train on the generated data
"""

svc = SVC(C = 1, kernel = 'linear') # Choosing best C we use k - fold cross validation
svc.fit(X,y)                        ## Partition into k "folds" each partition has chance to be validation, test, train

"""We can now plot the support vector classifier by calling the ${\tt plot\_svc()}$ function on the output of the call to ${\tt SVC()}$, as well as the data used in the call to ${\tt SVC()}$:
1. Create a plot with `plot_svc(svc, X, y)`
"""

plot_svc(svc, X, y)

"""The region of feature space that will be assigned to the −1 class is shown in
light blue, and the region that will be assigned to the +1 class is shown in
brown. The decision boundary between the two classes is linear (because we
used the argument ${\tt kernel="linear"}$).

The support vectors are plotted with stars
and the remaining observations are plotted as circles; we see here that there
are 13 support vectors. We can determine their identities as follows:

1. Use attributes `support_` and `support_vectors_` to see the data ID and data points
"""

svc.support_ # data indices for which data points are support vectors

svc.support_vectors_ #gives vectors for corresponding indices

"""What if we instead used a smaller value of the ${\tt C}$ parameter? Like 0.1.
1. Create a SVC instance called svc2 with parameter `C=0.1` and `kernel = 'linear'`
2. Train on the generated data
3. Create the plot of the SVC
"""

svc2 = SVC(C = 0.1, kernel = 'linear')
svc2.fit(X,y)

plot_svc(svc2, X, y) # notice this also gives num of support vectors

"""Now that a smaller value of the ${\tt c}$ parameter is being used, we obtain a
larger number of support vectors, because the margin is now **wider**.

The ${\tt sklearn.grid\_search}$ module includes a a function ${\tt GridSearchCV()}$ to perform cross-validation. In order to use this function, we pass in relevant information about the set of models that are under consideration. The
following command indicates that we want perform 10-fold cross-validation to compare SVMs with a linear
kernel, using a range of values of the cost parameter:
"""

from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'C' : [0.001, 0.01, 0.1, 1, 5, 10, 100]}]
clf = GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv = 10, scoring = 'accuracy') # cv = 10 ... ten-fold cross validation
clf.fit(X,y)                                                                              # scoring = 'accuracy' is parameter for what is best

"""We can easily access the cross-validation errors for each of these models by using attribute **cv_results_**"""

clf.cv_results_

"""The ${\tt GridSearchCV()}$ function stores the best parameters obtained, which can be accessed  by attribue **best_params_**"""

clf.best_params_ # if there is a tie sklearn returns first one in order of tie (left to right)

"""c=5 is best according to ${\tt GridSearchCV}$.

As usual, the ${\tt predict()}$ function can be used to predict the class label on a set of
test observations, at any given value of the cost parameter. Let's
generate a test data set:
"""

np.random.seed(1)
X_test = np.random.randn(20,2)
y_test = np.random.choice([-1,1], 20)
X_test[y_test == 1] = X_test[y_test == 1] -1

"""Now we predict the class labels of these test observations. Here we use the
best model obtained through cross-validation in order to make predictions:
1. Train an SVC called `svc2` with the best `C` value on X and y
2. Apply it to the test set, X_test and y_test, to see its perforamnce. You might use `score()` or `classification_report()` or `confusion_matrix`
"""

# earlier we saw C = 5 was best fit
svc2 = SVC(C = 5, kernel = 'linear')
svc2.fit(X,y)
svc2.score(X,y)

clf.best_estimator_ #WOW!!

clf.best_estimator_.score(X_test,y_test)

"""With this value of ${\tt c}$, 13 of the test observations are correctly
classified.

Now consider a situation in which the two classes are linearly separable.
Then we can find a separating hyperplane using the ${\tt svm()}$ function. First we'll give our simulated data a little nudge so that they are linearly separable:
"""

X_LS = X_test
y_LS = y_test
X_LS[y_LS == 1] = X_test[y_LS == 1] -1

plt.scatter(X_LS[:,0], X_LS[:,1], s=70, c=y_LS, cmap=mpl.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')

"""Now the observations are **just barely linearly** separable. We fit the support
vector classifier and plot the resulting hyperplane, using a very large value
of ${\tt cost}$ so that no observations are misclassified. Set $\tt C = 1e5$.
1. Create a SVC instance called `svc3` with parameter `C=1e5` and `kernel = 'linear'`
2. Train on the generated data X_LS and y_LS
3. Create the plot of the SVM
"""

svc3 = SVC(C = 1e5, kernel = 'linear') #bigger C - the more we care about loss
svc3.fit(X_LS, y_LS)
plot_svc(svc3, X_LS, y_LS)

"""No training errors were made and only three support vectors were used.
However, we can see from the figure that the margin is very narrow (because
the observations that are **not** support vectors, indicated as circles, are very close to the decision boundary). It seems likely that this model will perform
poorly on test data. Let's try a smaller value of ${\tt cost}$. Set $\tt C = 1$.

1. Create a SVC instance called `svc4` with parameter `C=1` and `kernel = 'linear'`
2. Train on the generated data X_LS and y_LS
3. Create the plot of the SVM
"""

svc4 = SVC(C = 1, kernel = 'linear')
svc4.fit(X_LS, y_LS)
plot_svc(svc4, X_LS, y_LS)

"""Using ${\tt cost=1}$, we misclassify a training observation, but we also obtain
a much wider margin and make use of five support vectors. It seems
likely that this model will perform better on test data than the model with
${\tt cost=1e5}$.

# Support Vector Machine

In order to fit an SVM using a **non-linear kernel**, we once again use the ${\tt SVC()}$
function. However, now we use a different value of the parameter kernel.
To fit an SVM with a polynomial kernel we use ${\tt kernel="poly"}$, and
to fit an SVM with a radial kernel we use ${\tt kernel="rbf"}$. In the former
case we also use the ${\tt degree}$ argument to specify a degree for the polynomial
kernel, and in the latter case we use ${\tt gamma}$ to specify a
value of $\gamma$ for the radial basis kernel.

Let's generate some data with a non-linear class boundary:
"""

from sklearn.model_selection import train_test_split

np.random.seed(8)
X = np.random.randn(200,2)
X[:100] = X[:100] +2
X[101:150] = X[101:150] -2
y = np.concatenate([np.repeat(-1, 150), np.repeat(1,50)])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=2)

plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=mpl.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2')

"""See how one class is kind of stuck in the middle of another class? This suggests that we might want to use a **radial kernel** in our SVM. Now let's fit
the training data using the ${\tt SVC()}$ function with a radial kernel, $\gamma = 1$  and $C =1$:
1. Create a SVC instance called `svm` with parameter `C=1`, `kernel = 'rbf'` and `gamma =1`
2. Train on the generated data X_train and y_train
3. Create the plot of the SVM with `plot_svc(svm, X_test, y_test)`
"""

svm = SVC(C=1, kernel = 'rbf', gamma = 1)
svm.fit(X_train, y_train)
plot_svc(svm, X_test, y_test)

"""Not too shabby! The plot shows that the resulting SVM has a decidedly non-linear
boundary. We can see from the figure that there are a fair number of training errors
in this SVM fit. If we increase the value of cost, we can reduce the number
of training errors. Set $C=100$.
1. Create a SVC instance called `svm2` with parameter `C=100`, `kernel = 'rbf'` and `gamma =1`
2. Train on the generated data X_train and y_train
3. Create the plot of the SVM with `plot_svc(svm2, X_test, y_test)`
"""

# Increasing C parameter, allowing more flexibility
svm2 = SVC(C = 100, kernel = 'rbf', gamma = 1)
svm2.fit(X_train,y_train)
plot_svc(svm2, X_test, y_test)

"""However, this comes at the price of a more irregular decision boundary that seems to be at risk of overfitting the data. We can perform cross-validation using ${\tt GridSearchCV()}$ to select the best choice of
$\gamma$ and cost for an SVM with a radial kernel. Set $C=[0.01, 0.1, 1, 10, 100]$ and $\gamma = [0.5, 1,2,3,4] $.
1. Create `tuned_parameters`
2. Use `GridSearchCV` to find the best combination of the parameters
"""

C = [0.01,0.1,1,10,100]
gamma = [0.5,1,2,3,4]
tuned_parameters = [{'C' : [0.01,0.1,1,10,100], 'gamma' : [0.5,1,2,3,4]}]
#GridSearchCV(SVC(kernel='linear'), tuned_parameters, cv = 10, scoring = 'accuracy')
clf = GridSearchCV(SVC(kernel = 'rbf'),tuned_parameters, cv = 10, scoring = 'accuracy')
clf.fit(X_train,y_train)

clf.cv_results_

clf.best_params_

clf.best_estimator_

clf.best_estimator_.score(X_test,y_test)

"""Therefore, the best choice of parameters involves ${\tt cost=10}$ and ${\tt gamma=0.5}$. We
can plot the resulting fit using the ${\tt plot\_svc()}$ function, and view the test set predictions for this model by applying the ${\tt predict()}$
function to the test data:
1. Create an SVC instance called `svm3` with the best combination of parameter C and gamma
2. Train on the generated data X_train and y_train
3. Create the plot of the SVM with `plot_svc(svm3, X_test, y_test)`
4. Output its score.
"""

svm3 = SVC(C = 10, kernel = 'rbf', gamma = 0.5)
svm3.fit(X_train, y_train)
plot_svc(svm3, X_test, y_test)

"""87% of test observations are correctly classified by this SVM. Not bad!"""
