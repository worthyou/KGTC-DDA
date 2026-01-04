import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch_geometric.nn import TransformerConv

class GraphIsomorphismLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, eps=0):
        super(GraphIsomorphismLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = Parameter(th.FloatTensor([eps]))
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, input, adj):
        h = (1 + self.eps) * input + th.spmm(adj, input)
        return self.mlp(h)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GIN(nn.Module):
    def __init__(self, features, nhid1, nhid2, dropout):
        super().__init__()
        self.gin1 = GraphIsomorphismLayer(features, nhid1)
        self.gin2 = GraphIsomorphismLayer(nhid1, nhid2)
        self.residual_linear = nn.Linear(features, nhid2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(nhid1)
        self.layer_norm2 = nn.LayerNorm(nhid2)

    def forward(self, x, adj):
        residual = x
        x = self.layer_norm1(F.relu(self.gin1(x, adj)))
        x = self.dropout(x)
        x = self.layer_norm2(F.relu(self.gin2(x, adj)))
        x = self.dropout(x)
        residual = self.residual_linear(residual)
        return x + residual

class GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4, dropout=0.3):
        super().__init__()
        self.trans1 = TransformerConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.trans2 = TransformerConv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.trans1(x, edge_index))
        x = self.dropout(x)
        x = self.trans2(x, edge_index)
        return x

class FGIN(nn.Module):
    def __init__(self, fdim_drug, fdim_disease, nhid1, nhid2, dropout, use_gin=True, use_transformer=True):
        super(FGIN, self).__init__()
        self.use_gin = use_gin
        self.use_transformer = use_transformer
        if use_gin:
            self.gin_drug = GIN(fdim_drug, nhid1, nhid2, dropout)
            self.gin_disease = GIN(fdim_disease, nhid1, nhid2, dropout)
        if use_transformer:
            self.trans_drug = GraphTransformer(fdim_drug, nhid1, nhid2, num_heads=4, dropout=dropout)
            self.trans_disease = GraphTransformer(fdim_disease, nhid1, nhid2, num_heads=4, dropout=dropout)
        self.fusion_drug = nn.Linear(2 * nhid2 if (use_gin and use_transformer) else nhid2, nhid2)
        self.fusion_disease = nn.Linear(2 * nhid2 if (use_gin and use_transformer) else nhid2, nhid2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_graph, drug_sim_feat, dis_graph, disease_sim_feat):
        drug_embed_gin = drug_sim_feat if not self.use_gin else self.gin_drug(drug_sim_feat, drug_graph)
        disease_embed_gin = disease_sim_feat if not self.use_gin else self.gin_disease(disease_sim_feat, dis_graph)
        drug_edge_index = drug_graph.coalesce().indices()
        dis_edge_index = dis_graph.coalesce().indices()
        drug_embed_trans = drug_sim_feat if not self.use_transformer else self.trans_drug(drug_sim_feat, drug_edge_index)
        disease_embed_trans = disease_sim_feat if not self.use_transformer else self.trans_disease(disease_sim_feat, dis_edge_index)

        if self.use_gin and self.use_transformer:
            drug_combined = th.cat([drug_embed_gin, drug_embed_trans], dim=-1)
            disease_combined = th.cat([disease_embed_gin, disease_embed_trans], dim=-1)
        elif self.use_gin:
            drug_combined = drug_embed_gin
            disease_combined = disease_embed_gin
        elif self.use_transformer:
            drug_combined = drug_embed_trans
            disease_combined = disease_embed_trans
        else:
            raise ValueError("At least one of GIN or GraphTransformer must be used.")

        drug_embed = F.relu(self.fusion_drug(drug_combined))
        disease_embed = F.relu(self.fusion_disease(disease_combined))
        return drug_embed, disease_embed

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=3, dropout=0.3):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        num_query = query.size(0)
        num_key = key_value.size(0)
        q = self.q_linear(query).view(num_query, self.num_heads, self.head_dim)
        k = self.k_linear(key_value).view(num_key, self.num_heads, self.head_dim)
        v = self.v_linear(key_value).view(num_key, self.num_heads, self.head_dim)
        scores = th.einsum('qhd,khd->hqk', q, k) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = th.einsum('hqk,khd->qhd', attn_weights, v)
        attn_output = attn_output.reshape(num_query, -1)
        output = self.layer_norm(self.out_linear(attn_output) + query)
        #output = self.out_linear(attn_output) + query
        return output, attn_weights

class MLP(nn.Module):
    def __init__(self, in_units, dropout_rate=0.3):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.lin1 = nn.Linear(2 * in_units, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, graph, drug_feat, dis_feat):
        with graph.local_scope():
            graph.nodes['drug'].data['h'] = drug_feat
            graph.nodes['disease'].data['h'] = dis_feat
            graph.apply_edges(udf_u_mul_e, etype='rate')
            out = graph.edges['rate'].data['m']
            out = F.relu(self.lin1(out))
            out = self.dropout(out)
            out = F.relu(self.lin2(out))
            out = self.dropout(out)
            out = self.lin3(out)
        return out

def udf_u_mul_e(edges):
    return {'m': th.cat([edges.src['h'], edges.dst['h']], dim=1)}