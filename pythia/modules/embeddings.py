# Copyright (c) Facebook, Inc. and its affiliates.
# TODO: Update kwargs with defaults
import os
import pickle
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pythia.modules.attention import AttentionLayer
from pythia.modules.layers import Identity
from pythia.utils.vocab import Vocab


class TextEmbedding(nn.Module):
    def __init__(self, emb_type, **kwargs):
        super(TextEmbedding, self).__init__()
        self.model_data_dir = kwargs.get("model_data_dir", None)
        self.embedding_dim = kwargs.get("embedding_dim", None)

        # Update kwargs here
        if emb_type == "identity":
            self.module = Identity()
            self.module.text_out_dim = self.embedding_dim
        elif emb_type == "vocab":
            self.module = VocabEmbedding(**kwargs)
            self.module.text_out_dim = self.embedding_dim
        elif emb_type == "preextracted":
            self.module = PreExtractedEmbedding(**kwargs)
        elif emb_type == "bilstm":
            self.module = BiLSTMTextEmbedding(**kwargs)
        elif emb_type == "attention":
            self.module = AttentionTextEmbedding(**kwargs)
        elif emb_type == "self_attn":
            self.module = SelfTextMultiHeadAttention(**kwargs)
        elif emb_type == "merger":
            self.module = SelfMergeAttention(**kwargs)
        elif emb_type == 'self_seq_att':
            self.module = SelfSeqAttention(**kwargs)
        elif emb_type == 'self_seq_att_2':
            self.module = SelfSeqAttention2(**kwargs)
        elif emb_type == "torch":
            vocab_size = kwargs["vocab_size"]
            embedding_dim = kwargs["embedding_dim"]
            self.module = nn.Embedding(vocab_size, embedding_dim)
            self.module.text_out_dim = self.embedding_dim
        else:
            raise NotImplementedError("Unknown question embedding '%s'" % emb_type)

        self.text_out_dim = self.module.text_out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class VocabEmbedding(nn.Module):
    def __init__(self, embedding_dim, vocab_params):
        self.vocab = Vocab(**vocab_params)
        self.module = self.vocab.get_embedding(nn.Embedding, embedding_dim)

    def forward(self, x):
        return self.module(x)


class BiLSTMTextEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        num_layers,
        dropout,
        bidirectional=False,
        rnn_type="GRU",
    ):
        super(BiLSTMTextEmbedding, self).__init__()
        self.text_out_dim = hidden_dim
        self.bidirectional = bidirectional

        if rnn_type == "LSTM":
            rnn_cls = nn.LSTM
        elif rnn_type == "GRU":
            rnn_cls = nn.GRU

        self.recurrent_encoder = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.recurrent_encoder(x)
        # Return last state
        if self.bidirectional:
            return out[:, -1]

        forward_ = out[:, -1, : self.num_hid]
        backward = out[:, 0, self.num_hid :]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        output, _ = self.recurrent_encoder(x)
        return output


class PreExtractedEmbedding(nn.Module):
    def __init__(self, out_dim, base_path):
        super(PreExtractedEmbedding, self).__init__()
        self.text_out_dim = out_dim
        self.base_path = base_path
        self.cache = {}

    def forward(self, qids):
        embeddings = []
        for qid in qids:
            embeddings.append(self.get_item(qid))
        return torch.stack(embeddings, dim=0)

    @lru_cache(maxsize=5000)
    def get_item(self, qid):
        return np.load(os.path.join(self.base_path, str(qid.item()) + ".npy"))


class AttentionTextEmbedding(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers, dropout, **kwargs):
        super(AttentionTextEmbedding, self).__init__()

        self.text_out_dim = hidden_dim * kwargs["conv2_out"]

        bidirectional = kwargs.get("bidirectional", False)

        self.recurrent_unit = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        conv1_out = kwargs["conv1_out"]
        conv2_out = kwargs["conv2_out"]
        kernel_size = kwargs["kernel_size"]
        padding = kwargs["padding"]

        self.conv1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        batch_size = x.size(0)

        self.recurrent_unit.flatten_parameters()
        # self.recurrent_unit.flatten_parameters()
        lstm_out, _ = self.recurrent_unit(x)  # N * T * hidden_dim
        lstm_drop = self.dropout(lstm_out)  # N * T * hidden_dim
        lstm_reshape = lstm_drop.permute(0, 2, 1)  # N * hidden_dim * T

        qatt_conv1 = self.conv1(lstm_reshape)  # N x conv1_out x T
        qatt_relu = self.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)  # N x conv2_out x T

        # Over last dim
        qtt_softmax = nn.functional.softmax(qatt_conv2, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature = torch.bmm(qtt_softmax, lstm_drop)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)

        # print(qtt_feature_concat.shape)

        return qtt_feature_concat


class SelfMergeAttention(nn.Module):
    def __init__(self, **kwargs):
        super(SelfMergeAttention, self).__init__()

        self.text_out_dim = kwargs["hidden_dim"] * kwargs["conv2_out"]

        self.m1 = AttentionTextEmbedding(**kwargs)
        self.m2 = SelfTextMultiHeadAttention(**kwargs)
        self.merger = nn.Linear(4096, 2048)

    def forward(self, x):
        x1 = self.m1(x)
        x2 = self.m2(x)

        x = torch.cat([x1, x2], 1)
        return self.merger(x)


# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
class SelfSeqAttention(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers, dropout, **kwargs):
        super(SelfSeqAttention, self).__init__()

        self.text_out_dim = hidden_dim * kwargs["conv2_out"]
        n_head = 4
        self.n_head = n_head

        self.in_pro = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1),
            nn.LeakyReLU(0.25),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1),
        )

        self.q_dim = 256
        self.v_dim = 512

        self.w_qs_1 = nn.Linear(embedding_dim, n_head * self.q_dim)
        self.w_ks_1 = nn.Linear(embedding_dim, n_head * self.q_dim)
        self.w_vs_1 = nn.Linear(embedding_dim, n_head * self.v_dim)
        nn.init.normal_(self.w_qs_1.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.q_dim)))
        nn.init.normal_(self.w_ks_1.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.q_dim)))
        nn.init.normal_(self.w_vs_1.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.v_dim)))

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.p_conv1 = nn.Conv1d(n_head * self.v_dim, embedding_dim, kernel_size=1)

        bidirectional = kwargs.get("bidirectional", False)

        self.recurrent_unit = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        conv1_out = kwargs["conv1_out"]
        conv2_out = kwargs["conv2_out"]
        kernel_size = kwargs["kernel_size"]
        padding = kwargs["padding"]

        self.conv1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.in_pro(x)
        x = x.permute(0, 2, 1).contiguous()

        b, t, c = x.shape
        x = x.view(b*t, -1)

        q = self.w_qs_1(x).view(b, t, self.n_head, self.q_dim)
        k = self.w_ks_1(x).view(b, t, self.n_head, self.q_dim)
        v = self.w_vs_1(x).view(b, t, self.n_head, self.v_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, t, self.q_dim) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, t, self.q_dim) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, t, self.v_dim) # (n*b) x lv x dv

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = F.softmax(attn, dim=2)
        x = torch.bmm(attn, v)

        x = x.view(self.n_head, b, t, self.v_dim)
        x = x.permute(1, 0, 3, 2).contiguous().view(b, -1, t)

        x = self.p_conv1(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.layer_norm_1(x)

        batch_size = x.size(0)

        self.recurrent_unit.flatten_parameters()
        # self.recurrent_unit.flatten_parameters()
        lstm_out, _ = self.recurrent_unit(x)  # N * T * hidden_dim
        lstm_drop = self.dropout(lstm_out)  # N * T * hidden_dim
        lstm_reshape = lstm_drop.permute(0, 2, 1)  # N * hidden_dim * T

        qatt_conv1 = self.conv1(lstm_reshape)  # N x conv1_out x T
        qatt_relu = self.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)  # N x conv2_out x T

        # Over last dim
        qtt_softmax = nn.functional.softmax(qatt_conv2, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature = torch.bmm(qtt_softmax, lstm_drop)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)

        # print(qtt_feature_concat.shape)

        return qtt_feature_concat

# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
class SelfSeqAttention2(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers, dropout, **kwargs):
        super(SelfSeqAttention2, self).__init__()

        self.text_out_dim = hidden_dim * kwargs["conv2_out"]
        n_head = 4
        self.n_head = n_head

        self.in_pro = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1),
            nn.LeakyReLU(0.25),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1),
        )

        self.q_dim = 256
        self.v_dim = 512

        self.w_qs_1 = nn.Linear(embedding_dim, n_head * self.q_dim)
        self.w_ks_1 = nn.Linear(embedding_dim, n_head * self.q_dim)
        self.w_vs_1 = nn.Linear(embedding_dim, n_head * self.v_dim)
        nn.init.normal_(self.w_qs_1.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.q_dim)))
        nn.init.normal_(self.w_ks_1.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.q_dim)))
        nn.init.normal_(self.w_vs_1.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.v_dim)))

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.p_conv1 = nn.Conv1d(n_head * self.v_dim, embedding_dim, kernel_size=1)

        self.w_qs_2 = nn.Linear(embedding_dim, n_head * self.q_dim)
        self.w_ks_2 = nn.Linear(embedding_dim, n_head * self.q_dim)
        self.w_vs_2 = nn.Linear(embedding_dim, n_head * self.v_dim)
        nn.init.normal_(self.w_qs_2.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.q_dim)))
        nn.init.normal_(self.w_ks_2.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.q_dim)))
        nn.init.normal_(self.w_vs_2.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.v_dim)))

        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.p_conv2 = nn.Conv1d(n_head * self.v_dim, embedding_dim, kernel_size=1)

        bidirectional = kwargs.get("bidirectional", False)

        self.recurrent_unit = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        conv1_out = kwargs["conv1_out"]
        conv2_out = kwargs["conv2_out"]
        kernel_size = kwargs["kernel_size"]
        padding = kwargs["padding"]

        self.conv1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.in_pro(x)
        x = x.permute(0, 2, 1).contiguous()

        b, t, c = x.shape
        x = x.view(b*t, -1)

        q = self.w_qs_1(x).view(b, t, self.n_head, self.q_dim)
        k = self.w_ks_1(x).view(b, t, self.n_head, self.q_dim)
        v = self.w_vs_1(x).view(b, t, self.n_head, self.v_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, t, self.q_dim) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, t, self.q_dim) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, t, self.v_dim) # (n*b) x lv x dv

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = F.softmax(attn, dim=2)
        x = torch.bmm(attn, v)

        x = x.view(self.n_head, b, t, self.v_dim)
        x = x.permute(1, 0, 3, 2).contiguous().view(b, -1, t)

        x = self.p_conv1(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.layer_norm_1(x)

        # Second
        x = x.view(b*t, -1)

        q = self.w_qs_2(x).view(b, t, self.n_head, self.q_dim)
        k = self.w_ks_2(x).view(b, t, self.n_head, self.q_dim)
        v = self.w_vs_2(x).view(b, t, self.n_head, self.v_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, t, self.q_dim) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, t, self.q_dim) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, t, self.v_dim) # (n*b) x lv x dv

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = F.softmax(attn, dim=2)
        x = torch.bmm(attn, v)

        x = x.view(self.n_head, b, t, self.v_dim)
        x = x.permute(1, 0, 3, 2).contiguous().view(b, -1, t)

        x = self.p_conv2(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.layer_norm_2(x)

        batch_size = x.size(0)

        self.recurrent_unit.flatten_parameters()
        # self.recurrent_unit.flatten_parameters()
        lstm_out, _ = self.recurrent_unit(x)  # N * T * hidden_dim
        lstm_drop = self.dropout(lstm_out)  # N * T * hidden_dim
        lstm_reshape = lstm_drop.permute(0, 2, 1)  # N * hidden_dim * T

        qatt_conv1 = self.conv1(lstm_reshape)  # N x conv1_out x T
        qatt_relu = self.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)  # N x conv2_out x T

        # Over last dim
        qtt_softmax = nn.functional.softmax(qatt_conv2, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature = torch.bmm(qtt_softmax, lstm_drop)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)

        # print(qtt_feature_concat.shape)

        return qtt_feature_concat

# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py
class SelfTextMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers, dropout, **kwargs):
        super(SelfTextMultiHeadAttention, self).__init__()

        self.text_out_dim = hidden_dim * kwargs["conv2_out"]
        n_head = 4
        self.n_head = n_head

        self.in_pro = nn.Sequential(
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1),
            nn.LeakyReLU(0.25),
            nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1),
        )

        self.q_dim = 256
        self.v_dim = 512

        self.w_qs_1 = nn.Linear(embedding_dim, n_head * self.q_dim)
        self.w_ks_1 = nn.Linear(embedding_dim, n_head * self.q_dim)
        self.w_vs_1 = nn.Linear(embedding_dim, n_head * self.v_dim)
        nn.init.normal_(self.w_qs_1.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.q_dim)))
        nn.init.normal_(self.w_ks_1.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.q_dim)))
        nn.init.normal_(self.w_vs_1.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.v_dim)))

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.fc11 = nn.Linear(n_head * self.v_dim, embedding_dim)

        self.w_qs_2 = nn.Linear(embedding_dim, n_head * self.q_dim)
        self.w_ks_2 = nn.Linear(embedding_dim, n_head * self.q_dim)
        self.w_vs_2 = nn.Linear(embedding_dim, n_head * self.v_dim)
        nn.init.normal_(self.w_qs_2.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.q_dim)))
        nn.init.normal_(self.w_ks_2.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.q_dim)))
        nn.init.normal_(self.w_vs_2.weight, mean=0, std=np.sqrt(2.0 / (embedding_dim + self.v_dim)))

        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        self.fc12 = nn.Linear(n_head * self.v_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim*14, 2048)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.in_pro(x)
        x = x.permute(0, 2, 1).contiguous()

        b, t, c = x.shape
        x = x.view(b*t, -1)

        q = self.w_qs_1(x).view(b, t, self.n_head, self.q_dim)
        k = self.w_ks_1(x).view(b, t, self.n_head, self.q_dim)
        v = self.w_vs_1(x).view(b, t, self.n_head, self.v_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, t, self.q_dim) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, t, self.q_dim) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, t, self.v_dim) # (n*b) x lv x dv

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = F.softmax(attn, dim=2)
        x = torch.bmm(attn, v)

        x = x.view(self.n_head, b, t, self.v_dim)
        x = x.permute(1, 2, 0, 3).contiguous().view(b, t, -1) # b x lq x (n*dv)

        x = F.relu(self.fc11(x), inplace=True)
        x = self.layer_norm_1(x)

        # Second layer
        b, t, c = x.shape
        x = x.view(b*t, -1)

        q = self.w_qs_2(x).view(b, t, self.n_head, self.q_dim)
        k = self.w_ks_2(x).view(b, t, self.n_head, self.q_dim)
        v = self.w_vs_2(x).view(b, t, self.n_head, self.v_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, t, self.q_dim) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, t, self.q_dim) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, t, self.v_dim) # (n*b) x lv x dv

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = F.softmax(attn, dim=2)
        x = torch.bmm(attn, v)

        x = x.view(self.n_head, b, t, self.v_dim)
        x = x.permute(1, 2, 0, 3).contiguous().view(b, t, -1) # b x lq x (n*dv)

        x = F.relu(self.fc12(x), inplace=True)
        x = self.layer_norm_2(x)

        x = x.view(b, -1)
        x = self.fc2(x)

        return x


class ImageEmbedding(nn.Module):
    """
    parameters:

    input:
    image_feat_variable: [batch_size, num_location, image_feat_dim]
    or a list of [num_location, image_feat_dim]
    when using adaptive number of objects
    question_embedding:[batch_size, txt_embeding_dim]

    output:
    image_embedding:[batch_size, image_feat_dim]


    """

    def __init__(self, img_dim, question_dim, **kwargs):
        super(ImageEmbedding, self).__init__()

        self.image_attention_model = AttentionLayer(img_dim, question_dim, **kwargs)
        self.out_dim = self.image_attention_model.out_dim

    def forward(self, image_feat_variable, question_embedding, image_dims, extra={}):
        # N x K x n_att
        attention = self.image_attention_model(
            image_feat_variable, question_embedding, image_dims
        )
        att_reshape = attention.permute(0, 2, 1)

        order_vectors = getattr(extra, "order_vectors", None)

        if order_vectors is not None:
            image_feat_variable = torch.cat(
                [image_feat_variable, order_vectors], dim=-1
            )
        tmp_embedding = torch.bmm(
            att_reshape, image_feat_variable
        )  # N x n_att x image_dim
        batch_size = att_reshape.size(0)
        image_embedding = tmp_embedding.view(batch_size, -1)

        return image_embedding, attention


class ImageFinetune(nn.Module):
    def __init__(self, in_dim, weights_file, bias_file):
        super(ImageFinetune, self).__init__()
        with open(weights_file, "rb") as w:
            weights = pickle.load(w)
        with open(bias_file, "rb") as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]

        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = nn.functional.relu(i2)
        return i3
