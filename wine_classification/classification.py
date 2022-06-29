import pdb
import csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
SEED = 2022
torch.manual_seed(SEED)


class FeedForwardNet(nn.Module):
    """
    Simple feed forward neural net.
    """

    def __init__(self, n_features, n_output):

        super(FeedForwardNet, self).__init__()
        self.n_features = n_features
        self.n_hid = self.n_features * 2
        self.n_output = n_output

        self.hid = nn.Linear(self.n_features, self.n_hid)
        self.output = nn.Linear(self.n_hid, self.n_output)

    def forward(self, input, labels=None):
        """
        input: tensor of inputs, of [batch_size, n_features]
        """

        logits = self.hid(input)
        logits = self.output(logits)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(logits, labels)
        else:
            return logits


def extract_data(filename):
    """
    Preprocess the data and split it into train and test sets.
    """

    features, labels = [], []
    n_pos, n_neg = 0, 0

    with open(filename) as f:
        reader = csv.reader(f, quotechar='"', delimiter=';')
        first_line = False
        for row in reader:
            if not first_line:
                first_line = True
                continue

            score = int(row[-1])
            # binarize the labels
            if score < 5:
                labels.append(0)
                n_neg += 1
            elif score > 6:
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


def train_nn(nn_model, train_features, train_labels, n_epochs=25):
    """
    Train the NN model and batch the data.
    """

    batch_size = 8
    train_features = torch.tensor(train_features, dtype=torch.float)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    train_dataset = TensorDataset(train_features, train_labels)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=0.01)
    optimizer.zero_grad()

    nn_model.train()
    for epoch in range(n_epochs):
        for batch in train_dataloader:
            data, labels = batch
            loss = nn_model(data, labels)

            # update the model parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'Finished training epoch {epoch}')

    return nn_model


def predictions_nn(nn_model, test_features):
    """
    Generate the models predictions for the NN model.
    """

    batch_size = 8
    test_features = torch.tensor(test_features, dtype=torch.float)
    test_dataset = TensorDataset(test_features)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)
    softmax = nn.Softmax(dim=1)
    preds, probs = [], []

    nn_model.eval()
    for batch in test_dataloader:
        with torch.no_grad():
            logits = nn_model(batch[0])
            batch_probs = softmax(logits)

        batch_preds = np.argmax(logits.numpy(), axis=1).tolist()
        batch_probs = batch_probs.numpy()
        preds.extend(batch_preds)
        probs.append(batch_probs)

    preds = np.array(preds)
    probs = np.concatenate(probs, axis=0)
    return preds, probs


def evaluate(predictions, test_labels):
    """
    Compute evaluation metrics on the test set.
    """

    predicted_labels = predictions['predicted_labels']
    predicted_probs = predictions['predicted_probs']

    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    auc = roc_auc_score(test_labels, predicted_probs[:, 1])

    return precision, recall, f1, auc


def main():
    """
    Main function that trains, performs inference, and evaluates performance.
    """

    filename = './winequality-white.csv'
    train_features, train_labels, test_features, test_labels = extract_data(filename)
    train_features, test_features = preprocess(train_features, test_features)

    predictions = {'svm': dict(), 'lr': dict(), 'nn': dict()}
    # Model training and inference.
    svm = SVC(probability=True)
    svm = svm.fit(train_features, train_labels)
    predictions['svm']['predicted_labels'] = svm.predict(test_features)
    predictions['svm']['predicted_probs'] = svm.predict_proba(test_features)

    lr = LogisticRegression()
    lr = lr.fit(train_features, train_labels)
    predictions['lr']['predicted_labels'] = lr.predict(test_features)
    predictions['lr']['predicted_probs'] = lr.predict_proba(test_features)

    nn_model = FeedForwardNet(train_features.shape[1], 2)
    nn_model = train_nn(nn_model, train_features, train_labels)
    nn_preds, nn_probs = predictions_nn(nn_model, test_features)
    predictions['nn']['predicted_labels'] = nn_preds
    predictions['nn']['predicted_probs'] = nn_probs

    # Perform evaluation.
    print('Performing model evaluation')
    models = ['svm', 'lr', 'nn']
    for model in models:
        precision, recall, f1, auc = evaluate(predictions[model], test_labels)
        print(f'For {model}:')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1: {f1}')
        print(f'AUC: {auc}')
        print('---------------------------------------')


if __name__ == '__main__':

    main()
