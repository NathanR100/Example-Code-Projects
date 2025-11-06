# -*- coding: utf-8 -*-
"""Lab8_Dimensionality_Reduction.ipynb"""

# Commented out IPython magic to ensure Python compatibility.
from IPython.display import Image
import numpy as np
# %matplotlib inline

"""# Unsupervised dimensionality reduction via principal component analysis

## The main steps behind principal component analysis
"""

Image(filename='pca.png', width=400)

"""These data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.

The attributes are

1) Alcohol

2) Malic acid

3) Ash

4) Alcalinity of ash

5) Magnesium

6) Total phenols

7) Flavanoids

8) Nonflavanoid phenols

9) Proanthocyanins

10) Color intensity

11) Hue

12) OD280/OD315 of diluted wines

13) Proline

In a classification context, this is a well posed problem with "well behaved" class structures. A good data set for first testing of a new classifier, but not very challenging.



"""

import pandas as pd

# df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
#                       'machine-learning-databases/wine/wine.data',
#                       header=None)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

df_wine = pd.read_csv('wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()

"""<hr>

Splitting the data into 70% training and 30% test subsets.
1. Please import train_test_split from sklearn.model_selection
2. Set y as 'Class label'
3. Set X as all other variables
4. Split the data into 70% training and 30% test subsets. Use y to stratify the spliting.
"""

from sklearn.model_selection import train_test_split

y = df_wine['Class label']
X = df_wine.drop(['Class label'], axis = 1)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3, stratify = y, random_state = 0)

"""Standardizing the data.
1. Please import StandardScaler from sklearn.preprocessing
2. Create a standard scaler instance called `sc`
3. Use `fit_transform()` on training data and name the resulting data as `X_train_std`
4. Use `transfrom()` on test data and name the resulting data as `X_test_std`
"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

"""---

**Note**

As remember from Chapter 3, the correct way is to re-use parameters from the training set if we are doing any kind of transformation -- the test set should basically stand for "new, unseen" data.

A common mistake is that some people are *not* re-using these parameters from the model training/building and standardize the new data "from scratch." Here's simple example to explain why this is a problem.

Let's assume we have a simple training set consisting of 3 samples with 1 feature (let's call this feature "length"):

- train_1: 10 cm -> class_2
- train_2: 20 cm -> class_2
- train_3: 30 cm -> class_1

mean: 20, std.: 8.2

After standardization, the transformed feature values are

- train_std_1: -1.21 -> class_2
- train_std_2: 0 -> class_2
- train_std_3: 1.21 -> class_1

Next, let's assume our model has learned to classify samples with a standardized length value < 0.6 as class_2 (class_1 otherwise). So far so good. Now, let's say we have 3 unlabeled data points that we want to classify:

- new_4: 5 cm -> class ?
- new_5: 6 cm -> class ?
- new_6: 7 cm -> class ?

If we look at the "unstandardized "length" values in our training datast, it is intuitive to say that all of these samples are likely belonging to class_2. However, if we standardize these by re-computing standard deviation and and mean, your classifier would incorrectly classify samples 6 as class 1.

- new_std_4: -1.21 -> class 2
- new_std_5: 0 -> class 2
- new_std_6: 1.21 -> class 1

mean: 6, std.: 0.82

However, if we use the parameters from your "training set standardization," we'd get the values:

- sample5: -18.37 -> class 2
- sample6: -17.15 -> class 2
- sample7: -15.92 -> class 2

The values 5 cm, 6 cm, and 7 cm are much lower than anything we have seen in the training set previously. Thus, it only makes sense that the standardized features of the "new samples" are much lower than every standardized feature in the training set.

---

## Principal component analysis (PCA) in scikit-learn

We are going to used $PCA$ function to to this. It is part of $decomposition$ in scikit-learn.
1. Import PCA from sklearn.decomposition
2. Create a PCA instance called `pca`
3. Apply `fit_transform()` on X_train_std to get a tranformed data called `X_train_pca`
"""

from sklearn.decomposition import PCA
pca = PCA()

X_train_pca = pca.fit_transform(X_train_std)

"""### Total and explained variance"""

import matplotlib.pyplot as plt
import numpy as np
plt.bar(range(1, 14), pca.explained_variance_ratio_, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid',
        label='cumulative explained variance')
plt.legend(loc='best')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()

"""### Feature transformation

We will use the first two components. The argument is n_components.

1. Create a PCA instance called `pca` with `n_components=2`
2. Apply `fit_transform()` on X_train_std to get a tranformed data called `X_train_pca`
3. Apply `transform()` on X_test_std to get a tranformed data called `X_test_pca`
"""

pca = PCA(n_components = 2)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

"""The following cell has codes for plotting decision regions. Do not change it and just run it!"""

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx],
                    label=cl)

"""Training logistic regression classifier using the first 2 principal components (the training data after transformation).
1. Import LogisticRegression from sklearn.linear_model
2. Create a logistic regression instance call `lr`
3. Train it on the PCA tranformed traning data (the one with the first two principal componenets)
"""

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('pca_logReg_train.png', dpi=300)
plt.show()

plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('pca_logReg_test.png', dpi=300)
plt.show()

"""<br>
<br>

# Supervised data compression via linear discriminant analysis

## Principal component analysis versus linear discriminant analysis
"""

Image(filename='lda.png', width=400)

"""## LDA via scikit-learn

We will use the supervised technique, LDA, for dimensionality reduction. Use the first two componenets.

1. Import LinearDiscriminantAnalysis from sklearn.discriminant_analysis
2. Create a LDA instance called `lda` with `n_components=2`
2. Apply `fit_transform(X_train_std, y_train)` on X_train_std to get a tranformed data called `X_train_lda`
3. Apply `transform()` on X_test_std to get a tranformed data called `X_test_lda`
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 2)

X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

"""Training logistic regression classifier using the first two discriminants.
1. Create a logistic regression instance call `lr`
2. Train it on the LDA tranformed traning data (the one with the first two discriminants)
"""

plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_09.png', dpi=300)
plt.show()

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_10.png', dpi=300)
plt.show()

plt.bar(range(1, len(lda.explained_variance_ratio_)+1), lda.explained_variance_ratio_, alpha=0.5, align='center',
        label='individual "discriminability"')
plt.step(range(1, len(lda.explained_variance_ratio_)+1), np.cumsum(lda.explained_variance_ratio_), where='mid',
        label='cumulative "discriminability"')
plt.ylabel('"discriminability" ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

"""<br>
<br>

# Using kernel principal component analysis for nonlinear mappings
"""

Image(filename='kernel_pca.png', width=500)

"""<br>
<br>

## Implementing a kernel principal component analysis in Python

### Example 1: Separating half-moon shapes
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, random_state=123)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)

plt.tight_layout()
# plt.savefig('images/05_12.png', dpi=300)
plt.show()

"""Apply the PCA only
1. Create a PCA instance called `scikit_pca` with `n_components=2`
2. Apply `fit_transform()` on X to get a tranformed data called `X_spca`

"""

scikit_pca = PCA(n_components = 2)
X_spca = scikit_pca.fit_transform(X)

# plot the first two and the first component
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],
              color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02,
              color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
plt.show()

"""Let's see the power of Kernel PCA.
1. Import `KernelPCA` from `sklearn.decomposition`
2. Create a Kernel PCA instance called `scikit_kpca` with `n_components=2, kernel='rbf', gamma=15`
3. Apply `fit_transform()` on X to get a tranformed data called `X_skernpca`
"""

from sklearn.decomposition import KernelPCA

scikit_kpca = KernelPCA(n_components=2)
X_skernpca = scikit_kpca.fit_transform(X)

# plot kernel pca

plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1],
            color='blue', marker='o', alpha=0.5)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()

"""<br>

### Example 2: Separating concentric circles
"""

from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
plt.tight_layout()
plt.show()

"""Apply PCA
1. Create a PCA instance called `scikit_pca` with `n_components=2`
2. Apply `fit_transform()` on X to get a tranformed data called `X_spca`
"""

scikit_pca = PCA(n_components =2)

X_spca = scikit_pca.fit_transform(X)

# plot the first two and the first component
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],
              color='red', marker='^', alpha=0.5)
ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],
              color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_spca[y == 0, 0], np.zeros((500, 1)) + 0.02,
              color='red', marker='^', alpha=0.5)
ax[1].scatter(X_spca[y == 1, 0], np.zeros((500, 1)) - 0.02,
              color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
plt.show()

"""Apply kernel PCA
1. Import `KernelPCA` from `sklearn.decomposition`
2. Create a Kernel PCA instance called `scikit_kpca` with `n_components=2, kernel='rbf', gamma=15`
3. Apply `fit_transform()` on X to get a tranformed data called `X_skernpca`
"""

from sklearn.decomposition import KernelPCA

scikit_kpca = KernelPCA(n_components = 2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)

# plot kernel pca

plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1],
            color='blue', marker='o', alpha=0.5)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()
