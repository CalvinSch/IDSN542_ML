'''
Calvin Schaul
IDSN 542, Fall 2025
cschaul@usc.edu
Homework 5
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
import pandas as pd

'''
UC Irvine League Letter Recognition Dataset
'''


def import_letter_data():
    return pd.read_csv("letter-recognition/letter-recognition.data")

def check_missing_data(data):
    missing_data = data.isnull().sum()
    print("Missing data in each column:\n", missing_data)

def add_labels_to_data(data):
    # The first column is the label
    column_names = ['lettr', 'x-box', 'y-box', 'width', 'height', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx']
    data.columns = column_names

def confirm_no_missing_data(data):
    if data.isnull().sum().sum() == 0:
        print("No missing data found.")
    else:
        print("There is missing data.")

def main():
    letter_data = import_letter_data() # Letter data

    check_missing_data(letter_data)
    
    add_labels_to_data(letter_data)

    print(letter_data.head())

    confirm_no_missing_data(letter_data)

    # Drop label column for features to separate "answer" from data
    X,y = letter_data.drop('lettr', axis=1), letter_data['lettr']

    # From datasource: use first 16000 for training, last 4000 for testing
    X_train, X_test = X[:16000], X[16000:]
    y_train, y_test = y[:16000], y[16000:]

    print("\nFirst row of training data:\n", X_train.iloc[0])


    # Check count each letter in training set
    print("Training set letter counts:\n", y_train.value_counts())
    print("Quick sanity check - number of letters found: ", len(y_train.value_counts()))

    # Multiclass classification test Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    print("Random Forest Classifier - First entry test prediction:", rf_classifier.predict(X_test[:1]))
    print("Expected letter for first test case:", y_test.iloc[0])

    # S K Fold cross-validation for Random Forest Classifier
    skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    for train_index, test_index in skfolds.split(X_train, y_train):
        clone_clf = clone(rf_classifier)
        X_train_folds = X_train.iloc[train_index]
        y_train_folds = y_train.iloc[train_index]
        X_test_fold = X_train.iloc[test_index]
        y_test_fold = y_train.iloc[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print("Random Forest Classifier accuracy for fold:", n_correct / len(y_pred))

    # Now do cross val predict to get predictions for all training data and calculate scores
    y_train_pred = cross_val_predict(rf_classifier, X_train, y_train, cv=3)
    print("Random Forest Classifier - Overall training set Precision:", precision_score(y_train, y_train_pred, average='macro'))
    print("Random Forest Classifier - Overall training set Recall:", recall_score(y_train, y_train_pred, average='macro'))
    print("Random Forest Classifier - Overall training set F1 Score:", f1_score(y_train, y_train_pred, average='macro'))
    
    # Generate precision-recall curve for macro average
    # Convert labels to binary (1 for 'A', 0 for all other letters)
    y_binary = (y_train == 'A').astype(int)
    
    # Get probability scores for letter 'A'
    y_scores = cross_val_predict(rf_classifier, X_train, y_binary, cv=3, method="predict_proba")
    y_scores_forest = y_scores[:, 1]  # Probability of class 1 ('A')
    
    # Calculate precision-recall curve for letter 'A'
    precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_binary, y_scores_forest)    
    
    plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2, label="Letter 'A'")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for Letter 'A'")
    plt.legend(loc="best")
    plt.ylim([0,1])
    plt.show()

    # ROC curve for letter 'A'
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_binary, y_scores_forest)
    plt.plot(fpr_forest, tpr_forest, linewidth=2, label="Letter 'A'")
    plt.plot([0,1], [0,1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve for Letter 'A'")
    plt.legend(loc="best")
    plt.show()

    # Scale data and show confusion matrix for Random Forest Classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_pred_scaled = cross_val_predict(rf_classifier, X_train_scaled, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred_scaled)
    print("Confusion Matrix for Random Forest Classifier:\n", conf_mx)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.title("Confusion Matrix for Random Forest Classifier")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Normalize confusion matrix
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)  # Fill diagonal with zeros
    plt.matshow(norm_conf_mx)
    plt.title("Normalized Confusion Matrix for Random Forest Classifier")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
      
main()