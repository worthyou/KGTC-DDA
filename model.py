from layers import *

class DrugDiseaseModel(nn.Module):
    def __init__(self, args, use_gin=True, use_transformer=True):
        super(DrugDiseaseModel, self).__init__()
        self.FGIN = FGIN(args.fdim_drug,
                         args.fdim_disease,
                         args.nhid1,
                         args.nhid2,
                         args.dropout,
                         use_gin=use_gin,
                         use_transformer=use_transformer
                         )
        self.cross_attn_drug = CrossAttention(embed_dim=args.gcn_out_units, num_heads=args.num_heads, dropout=args.dropout)
        self.cross_attn_disease = CrossAttention(embed_dim=args.gcn_out_units, num_heads=args.num_heads, dropout=args.dropout)
        
        self.MLP = MLP(in_units=args.gcn_out_units)

    def forward(self, dec_graph,
                drug_graph, drug_sim_feat,
                dis_graph, disease_sim_feat):
        drug_embed, disease_embed = self.FGIN(drug_graph, drug_sim_feat,
                                              dis_graph, disease_sim_feat)

        drug_feats, att_drug = self.cross_attn_drug(drug_embed, disease_embed)

        disease_feats, att_dis = self.cross_attn_disease(disease_embed, drug_embed)

        pred_ratings = self.MLP(dec_graph, drug_feats, disease_feats)
        
        return pred_ratings