import copy
import json
import os
import pickle
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import mean_absolute_error
from torch import optim
from torch.utils.data import Dataset, DataLoader
from model import CategoricalGraphAtt

class StockDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def create_dataloader(x, y, batch_size):
    dataset = StockDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":

    data_path = './datasets/sp500_data.pkl'
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    inner_edge = np.array(np.load("./datasets/inner_edge.npy"))
    outer_edge = np.array(np.load("./datasets/outer_edge.npy"))
    time_step = data["train"]["x1"].shape[-2]
    input_dim = data["train"]["x1"].shape[-1]
    num_weeks = data["train"]["x1"].shape[0]
    train_size = int(num_weeks * 0.2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beta = 0.1
    alpha = 0.1
    gamma = 0.1

    # convert data into torch dtype
    train_x = torch.Tensor(data["train"]["x1"]).float().to(device)
    inner_edge = torch.tensor(inner_edge.T, dtype=torch.int64).to(device)
    outer_edge = torch.tensor(outer_edge.T, dtype=torch.int64).to(device)

    # test data
    test_x = torch.Tensor(data["test"]["x1"]).float().to(device)

    # label data
    train_y = torch.Tensor(data["train"]["y_return ratio"]).float().to(device)
    test_y = torch.Tensor(data["test"]["y_return ratio"]).float().to(device)

    dataloader = create_dataloader(train_x, train_y, batch_size=16)
    # 建立
    hidden_dim = 32
    model = CategoricalGraphAtt(input_dim, time_step, hidden_dim, inner_edge, outer_edge, device).to(device)
    # initialize parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:%s" % pytorch_total_params)

    # optimizer & loss
    optimizer = optim.Adam(model.parameters(), weight_decay=0.1, lr=0.1)
    reg_loss_func = nn.L1Loss(reduction="none")
    cls_loss_func = nn.BCELoss(reduction="none")

    # save best model
    best_metric_IRR = None
    best_metric_MRR = None
    best_results_IRR = None
    best_results_MRR = None
    global_best_IRR = 999
    global_best_MRR = 0

    r_loss = torch.tensor([]).float().to(device)
    c_loss = torch.tensor([]).float().to(device)
    ra_loss = torch.tensor([]).float().to(device)
    for epoch in range(50):
        for week in range(num_weeks):
            model.train()  # prep to train model
            batch_x1 = train_x[week].to(device)
            batch_reg_y = train_y[week].view(-1, 1).to(device)
            batch_cls_y = train_y[week].view(-1, 1).to(device)
            reg_out, cls_out = model(batch_x1)
            reg_out, cls_out = reg_out.view(-1, 1), cls_out.view(-1, 1)

            # calculate loss
            reg_loss = reg_loss_func(reg_out, batch_reg_y)  # (target_size, 1)
            cls_loss = cls_loss_func(cls_out, batch_cls_y)
            rank_loss = torch.relu(
                -(reg_out.view(-1, 1) * reg_out.view(1, -1)) * (batch_reg_y.view(-1, 1) * batch_reg_y.view(1, -1)))
            c_loss = torch.cat((c_loss, cls_loss.view(-1, 1)))
            r_loss = torch.cat((r_loss, reg_loss.view(-1, 1)))
            ra_loss = torch.cat((ra_loss, rank_loss.view(-1, 1)))

            if (week + 1) % 1 == 0:
                cls_loss = beta * torch.mean(c_loss)
                reg_loss = alpha * torch.mean(r_loss)
                rank_loss = gamma * torch.sum(ra_loss)
                loss = reg_loss + rank_loss + cls_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                r_loss = torch.tensor([]).float().to(device)
                c_loss = torch.tensor([]).float().to(device)
                ra_loss = torch.tensor([]).float().to(device)
                if (week + 1) % 144 == 0:
                    print("REG Loss:%.4f CLS Loss:%.4f RANK Loss:%.4f  Loss:%.4f" % (
                    reg_loss.item(), cls_loss.item(), rank_loss.item(), loss.item()))
