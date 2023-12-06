import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGPooling, global_mean_pool, global_max_pool


class AttentionBlock(nn.Module):
    def __init__(self, time_step, dim):
        super(AttentionBlock, self).__init__()
        self.attention_matrix = nn.Linear(time_step, time_step)

    def forward(self, inputs):
        inputs_t = torch.transpose(inputs, 2, 1)  # (batch_size, input_dim, time_step)
        attention_weight = self.attention_matrix(inputs_t)
        attention_probs = F.softmax(attention_weight, dim=-1)
        attention_probs = torch.transpose(attention_probs, 2, 1)
        attention_vec = torch.mul(attention_probs, inputs)
        attention_vec = torch.sum(attention_vec, dim=1)
        return attention_vec, attention_probs


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim):
        super(SequenceEncoder, self).__init__()
        self.encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.attention_block = AttentionBlock(time_step, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.dim = hidden_dim

    def forward(self, seq):
        '''
        inp : torch.tensor (batch,time_step,input_dim)
        '''
        seq_vector, _ = self.encoder(seq)
        seq_vector = self.dropout(seq_vector)
        attention_vec, _ = self.attention_block(seq_vector)
        attention_vec = attention_vec.view(-1, 1, self.dim)  # prepare for concat
        return attention_vec


class CategoricalGraphAtt(nn.Module):
    def __init__(self, input_dim, time_step, hidden_dim, inner_edge, outer_edge, input_num, device):
        super(CategoricalGraphAtt, self).__init__()

        # basic parameters
        self.dim = hidden_dim
        self.input_dim = input_dim
        self.time_step = time_step
        self.inner_edge = inner_edge
        self.outer_edge = outer_edge
        self.input_num = input_num
        self.device = device

        # hidden layers
        self.pool_attention = AttentionBlock(20, hidden_dim)
        self.encoder = SequenceEncoder(input_dim, time_step, hidden_dim)
        self.cat_gat = GATConv(hidden_dim, hidden_dim)
        self.inner_gat = GATConv(hidden_dim, hidden_dim)
        self.weekly_attention = AttentionBlock(input_num, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)

        # output layer
        self.reg_layer = nn.Linear(hidden_dim, 1)
        self.cls_layer = nn.Linear(hidden_dim, 1)

    def forward(self, weekly_batch):
        print(f'weekly_batch.size = {weekly_batch.size()}')
        # x has shape (category_num, stocks_num, time_step, dim)

        weekly_att_vector = self.encoder(weekly_batch.view(-1, self.time_step, self.input_dim))  # (100,1,dim)
        print(f'weekly_att_vector.size = {weekly_att_vector.size()}')

        # inner graph interaction
        inner_graph_embedding = self.inner_gat(weekly_att_vector, self.inner_edge)
        inner_graph_embedding = inner_graph_embedding.view(5, 20, -1)

        # pooling
        weekly_att_vector = weekly_att_vector.view(5, 20, -1)
        category_vectors, _ = self.pool_attention(weekly_att_vector)  # torch.max(weekly_att_vector,dim=1)

        # use category graph attention
        category_vectors = self.cat_gat(category_vectors, self.outer_edge)  # (5,dim)
        category_vectors = category_vectors.unsqueeze(1).expand(-1, 20, -1)

        # fusion
        fusion_vec = torch.cat((weekly_att_vector, category_vectors, inner_graph_embedding), dim=-1)
        fusion_vec = torch.relu(self.fusion(fusion_vec))

        # output
        reg_output = self.reg_layer(fusion_vec)
        reg_output = torch.flatten(reg_output)
        cls_output = torch.sigmoid(self.cls_layer(fusion_vec))
        cls_output = torch.flatten(cls_output)

        return reg_output, cls_output

    def predict_toprank(self, test_data, device, top_k=5):
        y_pred_all_reg, y_pred_all_cls = [], []
        test_w1, test_w2, test_w3, test_w4 = test_data
        for idx, _ in enumerate(test_w2):
            batch_x1, batch_x2, batch_x3, batch_x4 = test_w1[idx].to(self.device), \
                                                     test_w2[idx].to(self.device), \
                                                     test_w3[idx].to(self.device), \
                                                     test_w4[idx].to(self.device)
            batch_weekly = [batch_x1, batch_x2, batch_x3, batch_x4][-self.input_num:]
            pred_reg, pred_cls = self.forward(batch_weekly)
            pred_reg, pred_cls = pred_reg.cpu().detach().numpy(), pred_cls.cpu().detach().numpy()
            y_pred_all_reg.extend(pred_reg.tolist())
            y_pred_all_cls.extend(pred_cls.tolist())
        return y_pred_all_reg, y_pred_all_cls

def MRR(test_y, pred_y, k=5):
    predict = pd.DataFrame([])
    predict["pred_y"] = pred_y
    predict["y"] = test_y

    predict = predict.sort_values("pred_y", ascending=False).reset_index(drop=True)
    predict["pred_y_rank_index"] = (predict.index) + 1
    predict = predict.sort_values("y", ascending=False)

    return sum(1 / predict["pred_y_rank_index"][:k])


def Precision(test_y, pred_y, k=5):
    predict = pd.DataFrame([])
    predict["pred_y"] = pred_y
    predict["y"] = test_y

    predict1 = predict.sort_values("pred_y", ascending=False)
    predict2 = predict.sort_values("y", ascending=False)
    correct = len(list(set(predict1["y"][:k].index) & set(predict2["y"][:k].index)))
    return correct / k


def IRR(test_y, pred_y, k=5):
    predict = pd.DataFrame([])
    predict["pred_y"] = pred_y
    predict["y"] = test_y

    predict1 = predict.sort_values("pred_y", ascending=False)
    predict2 = predict.sort_values("y", ascending=False)
    return sum(predict2["y"][:k]) - sum(predict1["y"][:k])


def Acc(test_y, pred_y):
    test_y = np.ravel(test_y)
    pred_y = np.ravel(pred_y)
    pred_y = (pred_y > 0) * 1
    acc_score = sum(test_y == pred_y) / len(pred_y)

    return acc_score