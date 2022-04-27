import csv
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
SEED = 2022
torch.manual_seed(SEED)


def extract_data(filename):
    """
    Preprocess the data and split it into train and test sets.
    """

    features, labels = [], []
    n_pos, n_neg = 0, 0

    with open(filename) as f:
        reader = csv.reader(f, quotechar='"', delimiter=';')
        for row in reader:
            score = int(row[-1])
            # binarize the labels
            if score < 5:
                labels.append(0)
                n_neg += 1
            elif score > 5:
                labels.append(1)
                n_pos += 1
            else:
                continue

            features.append([float(x) for x in row[:-1]])

    print(f'Total size of data: {len(features)}')
    print(f'Positive labels: {n_pos / (n_pos + n_neg)}')
    print(f'Negative labels: {n_neg / (n_pos + n_neg)}')

    features = np.array(features)
    labels = np.array(labels)

    # 9/10 training, 1/10 test
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=0.1, random_state=SEED)

    return train_features, train_labels, test_features, test_labels


def preprocess(train_features, test_features):
    """
    Perform some simple preprocessing and return the results. Subtract the mean and divide by the standard deviation.
    """

    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    return train_features, test_features


def build_nn(train_features, train_labels):
    """
    Build and train a simple feed forward neural network for binary classification.
    """

    pass


def evaluate(predictions, test_labels):
    """
    Compute evaluation metrics on the test set.
    """

    pass


def main():
    """
    Main function that trains, performs inference, and evaluates performance.
    """

    filename = './winequality-white.csv'
    train_features, train_labels, test_features, test_labels = extract_data(filename)
    train_features, test_features = preprocess(train_features, test_features)

    predictions = {'svm': dict(), 'lr': dict(), 'nn': dict()}
    # Model training and inference.
    svm = SVC()
    svm.fit(train_features, train_labels)
    predictions['svm']['predicted_labels'] = svm.predict(test_features)
    predictions['svm']['predicted_probs'] = svm.predict_proba(test_features)

    lr = LogisticRegression()
    lr.fit(train_features, train_labels)
    predictions['lr']['predicted_labels'] = lr.predict(test_features)
    predictions['lr']['predicted_probs'] = lr.predict_proba(test_features)

    nn = build_nn(train_features, train_labels)