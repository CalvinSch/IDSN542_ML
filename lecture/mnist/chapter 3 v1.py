from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score




#Fetch the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, parser="auto", as_frame=False)

print(mnist.keys())

X,y = mnist["data"], mnist["target"]
print(X.shape)  #70,000 images with 784 features each
print(y.shape)  #70,000 labels

'''
#Let's look at an instance's feature vector
some_digit = X[0]

some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation = "nearest")
plt.axis("off")
print(y[0])
plt.show()
'''

'''
#y is a bunch of text, not numbers. Let's convert to a number - better for computer
y = y.astype(np.uint8)

#Divide the 70,000 rows into the train set and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

print(len(X_train), len(X_test))  #Just to check we did it correctly
'''
'''
#Create two new objects that only have the images of 5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([some_digit]))
'''
''' 
skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred== y_test_fold)
    print(n_correct / len(y_pred)) #Supposed to print values around .95
'''

''' 
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
'''

'''
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

print(confusion_matrix(y_train_5, y_train_pred))
'''


#print("Precision:",precision_score(y_train_5, y_train_pred))
#print("Recall:",recall_score(y_train_5, y_train_pred))