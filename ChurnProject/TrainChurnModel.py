import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import opendatasets as od
import argparse

# Download the data
od.download('https://www.kaggle.com/datasets/blastchar/telco-customer-churn')

# Data Prep

def get_data(file):
    return pd.read_csv(file)

def lower_case(df):
    df.columns = df.columns.str.lower()
    return df

def get_formatted_data(file):
    df = lower_case(get_data(file))
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')

    # Removing entries with no values
    df = df[df['totalcharges'].isnull()==False]

    # Make DV Numeric
    df['churn'] = (df['churn']=='Yes') * 1

    # Partitioning Data into Train and Test Sets
    n = len(df)
    index = np.arange(0, n)
    rng = np.random.default_rng()
    rng.shuffle(index)
    df_train = df.iloc[index[0: 4 * n // 5]]
    df_test = df.iloc[index[4 * n // 5:]]
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    return df_train, df_test

def get_vectorized_data(file, features):
    df_train, df_test = get_formatted_data('churn_data/churn.csv')
    dv = DictVectorizer(sparse=False)

    train_dict = df_train[features].to_dict(orient='records')
    test_dict = df_test[features].to_dict(orient='records')

    X_train = dv.fit_transform(train_dict)
    X_test = dv.fit_transform(test_dict)
    Y_train = df_train['churn'].values
    Y_test = df_test['churn'].values
    return X_train, Y_train, X_test, Y_test, dv

# MLP Model

import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.lin1 = nn.Linear(45, nodes)
        self.lin2 = nn.Linear(nodes, nodes)
        self.lin3 = nn.Linear(nodes, 30)
        self.lin4 = nn.Linear(30, 1)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.drop = nn.Dropout(.4)
    def forward(self, x):
        x = self.drop(self.act(self.lin1(x)))
        x = self.drop(self.act(self.lin2(x)))
        x = self.drop(self.act(self.lin3(x)))
        return self.lin4(x)
    def predict_proba(self, x):
        '''For use with the plot_eval_metrics function and its sub functions'''
        x = torch.from_numpy(x).type(torch.float32)
        probs = self.sig(self.lin4(self.act(self.lin3(self.act(self.lin2(self.act(self.lin1(x))))))))
        return probs.detach().numpy()

class SmallModel(torch.nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.lin1 = nn.Linear(45, 1)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return self.lin1(x)
    def predict_proba(self, x):
        '''For parity with sklearn's LogisticRegression Class'''
        x = torch.from_numpy(x).type(torch.float32)
        probs = self.sig(self.forward(x))
        return probs.detach().numpy()

def train_epoch(model, batches, target, optimizer, device, criterion):
    running_loss = 0
    for i, batch in enumerate(batches):
        batch = batch.to(device)
        pred = model(batch).squeeze()
        loss = criterion(pred, target[i].to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item()
    return running_loss

def evaluate(model, batches, target, device, criterion):
    running_loss = 0
    for i, batch in enumerate(batches):
        batch = batch.to(device)
        pred = model(batch).squeeze()
        loss = criterion(pred, target[i].to(device))
        running_loss += loss.item()
    return running_loss

def batchify(data, batch_size):
    num_batches = data.shape[0] // batch_size
    batch_list = [data[batch_size * i: batch_size * (i + 1)] for i in range(num_batches)]
    return batch_list


def train(model, optimizer, num_epochs, X_train, Y_train, X_test, Y_test, batch_size, device='cpu', new_lr=None):
    batch_size = 64
    criterion = nn.BCEWithLogitsLoss()
    if new_lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    train_data = torch.from_numpy(X_train).type(torch.float32)
    train_targets = torch.from_numpy(Y_train).type(torch.float32)
    
    train_batches = batchify(train_data, batch_size)
    train_targ_batches = batchify(train_targets, batch_size)
    
    test_data = torch.from_numpy(X_test).type(torch.float32)
    test_targets = torch.from_numpy(Y_test).type(torch.float32)
    
    test_batches = batchify(test_data, batch_size)
    test_targ_batches = batchify(test_targets, batch_size)

    for i in range(num_epochs):
        epoch_loss = train_epoch(model, train_batches, train_targ_batches, optimizer, device, criterion)
        eval_loss = evaluate(model, test_batches, test_targ_batches, device, criterion)
        if i % 50 == 0:
            print(f'epoch {i} | train loss: {epoch_loss:.2f} | test loss: {eval_loss:.3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logit', metavar='filename', dest='logit_filename', default='logit_model')
    parser.add_argument('--mlp', metavar='filename', dest='mlp_filename', default='mlp_model')
    parser.add_argument('--epochs', metavar='integer', dest='num_epochs', type=int, default=500)
    args = parser.parse_args()
    print(args)
    features = ['gender', 'seniorcitizen', 'partner', 'dependents',
            'tenure', 'phoneservice', 'multiplelines', 'internetservice',
            'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
            'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
            'paymentmethod', 'monthlycharges', 'totalcharges']

    X_train, Y_train, X_test, Y_test, dv = get_vectorized_data('churn_data/churn.csv', features)

    # Train Logit Model
    print("Training Logit Model")
    logit = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    logit.fit(X_train, Y_train)

    # Save Logit Model 
    with open(args.logit_filename, 'wb') as file:
        pickle.dump((dv, logit), file)

    # Initialize and Train MLP Model
    mlp = Model(600).to('cpu')
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=0.0001)
    print(f'Training MLP Model for {args.num_epochs} Epochs')
    train(mlp, optimizer, args.num_epochs, X_train, Y_train, X_test, Y_test, batch_size=128, device='cpu')

    # Save MLP Model
    with open(args.mlp_filename, 'wb') as file:
        torch.save(mlp, file)
