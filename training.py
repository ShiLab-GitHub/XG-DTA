import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import pickle
from sklearn.model_selection import KFold

from config import DGSDTAModelConfig
from model import DGSDTAModel
from dataset import DGSDTADataset
from utils import *


def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        graph, seq_embed, seq_mask = data
        graph = graph.to(device)
        seq_embed = seq_embed.to(device)
        seq_mask = seq_mask.to(device)

        optimizer.zero_grad()
        output = model(graph, seq_embed)
        loss = loss_fn(output, graph.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(graph.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Making predictions for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for graph, seq_embed, seq_mask in loader:
            graph = graph.to(device)
            seq_embed = seq_embed.to(device)
            seq_mask = seq_mask.to(device)

            output = model(graph, seq_embed)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, graph.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def train_init(dataset='davis', pretrain=None):
    with open('data/seq2path_prot_albert.pickle', 'rb') as handle:
        seq2path = pickle.load(handle)
    with open('data/smile_graph.pickle', 'rb') as f:
        smile2graph = pickle.load(f)
    cuda_name = CUDA
    train_data = DGSDTADataset('data/{}_train.csv'.format(dataset), smile2graph, seq2path)
    test_data = DGSDTADataset('data/{}_test.csv'.format(dataset), smile2graph, seq2path)

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')

    config = DGSDTAModelConfig()
    graphNet = config['graphNet']
    model = DGSDTAModel(config).to(device)
    if pretrain:
        print("Using pretrain model {}".format(pretrain))
        state_dict = torch.load(pretrain)
        model.load_state_dict(state_dict)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    return model, device, train_data, test_data, loss_fn, optimizer, graphNet, dataset


def train_fold(model, device, train_data, test_data, loss_fn, optimizer, fold_idx):
    model_file_name = '111111model_' + graphNet + '_' + dataset + '_fold' + str(fold_idx + 1) + '.model'
    result_file_name = 'result_' + graphNet + '_' + dataset + '_fold' + str(fold_idx + 1) + '.csv'

    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1)
        G, P = predicting(model, device, test_loader)
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
        if ret[1] < best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_epoch = epoch + 1
            best_mse = ret[1]
            best_ci = ret[-1]
            print('RMSE improved at epoch {}, fold {}: Best MSE: {}, Best CI: {}'.format(best_epoch, fold_idx + 1,
                                                                                         best_mse, best_ci))
        else:
            print('No improvement since epoch {}, fold {}: Best MSE: {}, Best CI: {}'.format(best_epoch, fold_idx + 1,
                                                                                             best_mse, best_ci))


def run_cross_validation(model, device, train_data, test_data, loss_fn, optimizer, num_folds=5, num_runs=1):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_idx = 0
    for train_index, val_index in kf.split(train_data):
        train_fold_data = torch.utils.data.Subset(train_data, train_index)
        val_fold_data = torch.utils.data.Subset(train_data, val_index)

        print("Running fold {}...".format(fold_idx + 1))
        for run_idx in range(num_runs):
            model_file_name = '8.29model_' + graphNet + '_' + dataset + '_fold' + str(fold_idx + 1) + '_run' + str(
                run_idx + 1) + '.model'
            result_file_name = '8.29result_' + graphNet + '_' + dataset + '_fold' + str(fold_idx + 1) + '_run' + str(
                run_idx + 1) + '.csv'

            model.load_state_dict(torch.load(model_file_name)) if os.path.exists(model_file_name) else None

            train_fold(model, device, train_fold_data, val_fold_data, loss_fn, optimizer, fold_idx)
            # 保存模型的状态字典
            model_state_dict = model.state_dict()
            torch.save(model_state_dict, model_file_name)

        fold_idx += 1


TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.0004
LOG_INTERVAL = 100
NUM_EPOCHS = 500
CUDA = 'cuda:0'

pretrain = False
model, device, train_data, test_data, loss_fn, optimizer, graphNet, dataset = train_init(pretrain=pretrain)
print('Learning rate:', LR)
print('Epochs:', NUM_EPOCHS)

run_cross_validation(model, device, train_data, test_data, loss_fn, optimizer, num_folds=5, num_runs=1)
