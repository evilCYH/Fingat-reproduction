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
    def __init__(self, input_dim, time_step, hidden_dim, inner_edge, outer_edge, input_num, use_gru, device):
        super(CategoricalGraphAtt, self).__init__()

        # basic parameters
        self.dim = hidden_dim
        self.input_dim = input_dim
        self.time_step = time_step
        self.inner_edge = inner_edge
        self.outer_edge = outer_edge
        self.input_num = input_num
        self.use_gru = use_gru
        self.device = device

        # hidden layers
        # self.pool_attention = AttentionBlock(25,hidden_dim)
        if self.use_gru:
            self.weekly_encoder = nn.GRU(hidden_dim, hidden_dim)
        self.encoder_list = nn.ModuleList([SequenceEncoder(input_dim, time_step, hidden_dim) for _ in range(input_num)])
        self.cat_gat = GATConv(hidden_dim, hidden_dim)
        self.inner_gat = GATConv(hidden_dim, hidden_dim)
        self.weekly_attention = AttentionBlock(input_num, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)

        # output layer
        self.reg_layer = nn.Linear(hidden_dim, 1)
        self.cls_layer = nn.Linear(hidden_dim, 1)

    def forward(self, weekly_batch):
        # x has shape (category_num, stocks_num, time_step, dim)
        # print(f'weekly batch[i] size = {weekly_batch[0].size()}')  # torch.Size([475, 7, 30])
        weekly_embedding = self.encoder_list[0](weekly_batch[0].view(-1, self.time_step, self.input_dim))
        # print(f'weekly_embedding size = {weekly_embedding.size()}')    # torch.Size([475, 1, 64])

        # calculate embeddings for the rest of weeks
        for week_idx in range(1, self.input_num):
            weekly_inp = weekly_batch[week_idx]
            weekly_inp = weekly_inp.view(-1, self.time_step, self.input_dim)
            week_stock_embedding = self.encoder_list[week_idx](weekly_inp)
            weekly_embedding = torch.cat((weekly_embedding, week_stock_embedding), dim=1)
        # print(f'after concat weekly_embedding size = {weekly_embedding.size()}')  # torch.Size([475, 3, 64])

        # merge weeks
        if self.use_gru:
            weekly_embedding, _ = self.weekly_encoder(weekly_embedding)
        weekly_att_vector, _ = self.weekly_attention(weekly_embedding)
        # print(f'weekly_att_vector size = {weekly_att_vector.size()}')  # torch.Size([475, 64])

        # inner graph interaction
        inner_graph_embedding = self.inner_gat(weekly_att_vector, self.inner_edge)
        # print(f'inner_graph_embedding size = {inner_graph_embedding.size()}')  # torch.Size([475, 64])

        # pooling
        start_index = 0
        category_vectors_list = []
        for i in range(len(len_array)):
            end_index = start_index + len_array[i]
            sector_graph_embedding = inner_graph_embedding[start_index:end_index, :].unsqueeze(0)
            pool_attention = AttentionBlock(len_array[i], self.dim).to(device)
            category_vectors, _ = pool_attention(sector_graph_embedding)  # ([1, 64])
            category_vectors_list.append(category_vectors)
            start_index = end_index

        category_vectors = torch.cat(category_vectors_list, dim=0)  # torch.max(weekly_att_vector,dim=1)
        # print(f'category_vectors size = {category_vectors.size()}')  # torch.Size([19, 64])

        # use category graph attention
        category_vectors = self.cat_gat(category_vectors, self.outer_edge)  # (5,dim)
        # print(f'after sector gat category_vectors size = {category_vectors.size()}')  # torch.Size([19, 64])

        intra_graph_embedding_list = []
        for i in range(category_vectors.size()[0]):
            gat_category_vectors = category_vectors[i:i + 1, :]
            for j in range(len_array[i]):
                intra_graph_embedding_list.append(gat_category_vectors)
        intra_graph_embedding = torch.cat(intra_graph_embedding_list, dim=0)
        # print(f'intra_graph_embedding size = {intra_graph_embedding.size()}')   # torch.Size([475, 64])

        # fusion
        fusion_vec = torch.cat((weekly_att_vector, inner_graph_embedding, intra_graph_embedding), dim=-1)
        # print(f'cat fusion_vec size = {fusion_vec.size()}')  # torch.Size([475, 192])

        fusion_vec = self.fusion(fusion_vec)
        # print(f'linear fusion_vec size = {fusion_vec.size()}')  # torch.Size([475, 64])

        fusion_vec = torch.relu(fusion_vec)
        # print(f'relu fusion_vec size = {fusion_vec.size()}')  # torch.Size([475, 64])

        # output
        reg_output = self.reg_layer(fusion_vec)
        # print(f'reg_output size = {reg_output.size()}')  # torch.Size([475, 1])
        reg_output = torch.flatten(reg_output)
        # print(f'flatten reg_output size = {reg_output.size()}')  # torch.Size([475])

        cls_output = torch.sigmoid(self.cls_layer(fusion_vec))
        cls_output = torch.flatten(cls_output)

        return reg_output, cls_output

    def predict_toprank(self, test_data, device, top_k=5):
        y_pred_all_reg, y_pred_all_cls = [], []
        test_w1, test_w2, test_w3 = test_data
        for idx, _ in enumerate(test_w2):
            batch_x1, batch_x2, batch_x3 = test_w1[idx].to(self.device), \
                test_w2[idx].to(self.device), \
                test_w3[idx].to(self.device)
            batch_weekly = [batch_x1, batch_x2, batch_x3]
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