import os
import dgl
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import torch as th
import numpy as np

from sklearn.model_selection import KFold

_paths = {
    'Fdataset': './data/Fdataset/modified_dataset.mat',
    'Cdataset': './data/Cdataset/Cdataset.mat',
    'lrssl': './data/lrssl'
}

def normalize(mx, symmetric=False):
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv = np.where(rowsum > 0, 1.0 / rowsum, 0.0)
    if symmetric:
        r_inv_sqrt = np.sqrt(r_inv)
        r_mat_inv = sp.diags(r_inv_sqrt)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    else:
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)

def to_etype_name(rating):
    return str(rating).replace('.', '_')

def knn_graph(disMat, k):
    k_neighbor = np.argpartition(-disMat, kth=k, axis=1)[:, :k]
    row_index = np.arange(k_neighbor.shape[0]).repeat(k_neighbor.shape[1])
    col_index = k_neighbor.reshape(-1)
    edges = np.array([row_index, col_index]).astype(
        int).T
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(disMat.shape[0], disMat.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj

class DataLoader(object):
    def __init__(self,
                 name,
                 device,
                 symm=True,
                 k=4):
        self._name = name
        self._device = device
        self._symm = symm
        self.num_neighbor = k
        self._path = _paths.get(self._name, None)
        print(f"Loading dataset: {self._name} from {self._path}...")
        self._dir = os.path.join(_paths[self._name])
        self.cv_data_dict = self._load_drug_data(self._path, self._name)

        self._generate_graph_data()

    def _load_drug_data(self, file_path, data_name):
        association_matrix = None
        if data_name in ['Fdataset', 'Cdataset']:
            data = sio.loadmat(file_path)
            association_matrix = data['didr'].T

            print(f"关联矩阵形状: {association_matrix.shape}")
            num_associations = np.count_nonzero(association_matrix)
            print(f"药物-疾病关联数: {num_associations}")
            self.drug_structure = data['drug'].astype(np.float32)
            self.disease_ps = data['disease'].astype(np.float32)

            base_path = f"./data/{data_name}/"
            self.drug_glp = pd.read_csv(base_path + 'DrugGIP_new.csv', index_col=0).values.astype(np.float32)
            self.disease_glp = pd.read_csv(base_path + 'DiseaseGIP.csv', index_col=0).values.astype(np.float32)

            assert self.drug_structure.shape == self.drug_glp.shape, f"药物特征维度不匹配: {self.drug_structure.shape} vs {self.drug_glp.shape}"
            assert self.disease_ps.shape == self.disease_glp.shape, f"疾病特征维度不匹配: {self.disease_ps.shape} vs {self.disease_glp.shape}"

            self.drug_sim_features = (self.drug_structure + self.drug_glp) / 2
            self.disease_sim_features = (self.disease_ps + self.disease_glp) / 2

        elif data_name in ['lrssl']:
            data = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'), index_col=0, delimiter='\t')
            association_matrix = data.values
            print(f"关联矩阵形状: {association_matrix.shape}")
            num_associations = np.count_nonzero(association_matrix)
            print(f"药物-疾病关联数: {num_associations}")

            self.drug_structure = pd.read_csv(os.path.join(file_path, 'drug_chemical.txt'), index_col=0,delimiter='\t').values.astype(np.float32)
            self.disease_ps = pd.read_csv(os.path.join(file_path, 'DiseasePS.txt'), index_col=0,delimiter='\t').values.astype(np.float32)

            base_path = f"./data/{data_name}/"
            self.drug_glp = pd.read_csv(base_path + 'DrugGIP_new.csv', index_col=0).values.astype(np.float32)
            self.disease_glp = pd.read_csv(base_path + 'DiseaseGIP.csv', index_col=0).values.astype(np.float32)

            assert self.drug_structure.shape == self.drug_glp.shape, f"药物特征维度不匹配: {self.drug_structure.shape} vs {self.drug_glp.shape}"
            assert self.disease_ps.shape == self.disease_glp.shape, f"疾病特征维度不匹配: {self.disease_ps.shape} vs {self.disease_glp.shape}"
            
            self.drug_sim_features = (self.drug_structure + self.drug_glp) / 2
            self.disease_sim_features = (self.disease_ps + self.disease_glp) / 2

        self._num_drug = association_matrix.shape[0]
        self._num_disease = association_matrix.shape[1]

        kfold = KFold(n_splits=10, shuffle=True, random_state=1024)
        pos_row, pos_col = np.nonzero(association_matrix)
        neg_row, neg_col = np.nonzero(1 - association_matrix)
        assert len(pos_row) + len(neg_row) == np.prod(association_matrix.shape)
        
        self.num_pos_samples = len(pos_row)
        self.num_neg_samples = len(neg_row)
        
        cv_num = 0
        cv_data = {}

        for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                               kfold.split(neg_row)):
           train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
           train_pos_values = [1] * len(train_pos_edge[0])
           train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
           train_neg_values = [0] * len(train_neg_edge[0])

           test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
           test_pos_values = [1] * len(test_pos_edge[0])
           test_neg_edge = np.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])
           test_neg_values = [0] * len(test_neg_edge[0])

           train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
           test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)

           train_values = np.concatenate([train_pos_values, train_neg_values])
           test_values = np.concatenate([test_pos_values, test_neg_values])

           train_data = {
               'drug_id': train_edge[0],
               'disease_id': train_edge[1],
               'values': train_values
           }
           train_data_info = pd.DataFrame(train_data, index=None)

           test_data = pd.DataFrame({
               'drug_id': test_edge[0],
               'disease_id': test_edge[1],
               'values': test_values
           })
           test_data_info = pd.DataFrame(test_data, index=None)

           values = np.unique(train_values)
           cv_data[cv_num] = [train_data_info, test_data_info, values]
           cv_num += 1

        return cv_data

    def _generate_graph_data(self):
        self.data_cv = {}
        for cv in range(0, 10):
            self.train_data, self.test_data, self.values = self.cv_data_dict[cv]

            shuffled_idx = np.random.permutation(self.train_data.shape[0])
            self.train_rel_info = self.train_data.iloc[shuffled_idx[::]]
            self.test_rel_info = self.test_data
            self.possible_rel_values = self.values


            train_drug_ids = sorted(self.train_rel_info['drug_id'].unique().tolist())
            train_disease_ids = sorted(self.train_rel_info['disease_id'].unique().tolist())


            drug_graph, disease_graph = self._generate_feat_graph(train_drug_ids, train_disease_ids)


            train_pairs, train_values = self._generate_pair_value(self.train_rel_info)
            test_pairs, test_values = self._generate_pair_value(self.test_rel_info)


            self.train_enc_graph = self._generate_enc_graph(train_pairs, train_values, add_support=True,
                                                            drug_graph=drug_graph,
                                                            disease_graph=disease_graph)

            self.train_dec_graph = self._generate_dec_graph(train_pairs)
            self.train_truths = th.FloatTensor(train_values)


            self.test_enc_graph = self._generate_enc_graph(test_pairs, test_values, add_support=False,
                                                           drug_graph=drug_graph,
                                                           disease_graph=disease_graph)

            self.test_dec_graph = self._generate_dec_graph(test_pairs)
            self.test_truths = th.FloatTensor(test_values)


            self.data_cv[cv] = {'train': [self.train_enc_graph, self.train_dec_graph, self.train_truths],
                                'test': [self.test_enc_graph, self.test_dec_graph, self.test_truths],
                                'drug_graph': drug_graph,
                                'disease_graph': disease_graph,
                                'train_drug_ids': train_drug_ids,
                                'train_disease_ids': train_disease_ids
                                }
        return self.data_cv

    def _generate_feat_graph(self, train_drug_ids, train_disease_ids):
        drug_sim_submatrix = self.drug_sim_features[train_drug_ids][:, train_drug_ids]
        if sp.issparse(drug_sim_submatrix):
            drug_sim_submatrix = drug_sim_submatrix.toarray()
        elif not isinstance(drug_sim_submatrix, np.ndarray):
            drug_sim_submatrix = drug_sim_submatrix.values

        drug_num_neighbor = min(self.num_neighbor, drug_sim_submatrix.shape[0])
        drug_adj = knn_graph(drug_sim_submatrix, drug_num_neighbor)
        drug_graph = normalize(drug_adj + sp.eye(drug_adj.shape[0]))
        drug_graph = sparse_mx_to_torch_sparse_tensor(drug_graph)

        disease_sim_submatrix = self.disease_sim_features[train_disease_ids][:, train_disease_ids]
        if sp.issparse(disease_sim_submatrix):
            disease_sim_submatrix = disease_sim_submatrix.toarray()
        elif not isinstance(disease_sim_submatrix, np.ndarray):
            disease_sim_submatrix = disease_sim_submatrix.values

        disease_num_neighbor = min(self.num_neighbor, disease_sim_submatrix.shape[0])
        disease_adj = knn_graph(disease_sim_submatrix, disease_num_neighbor)
        disease_graph = normalize(disease_adj + sp.eye(disease_adj.shape[0]))
        disease_graph = sparse_mx_to_torch_sparse_tensor(disease_graph)

        return drug_graph, disease_graph
    @staticmethod
    def _generate_pair_value(rel_info):
        rating_pairs = (np.array([ele for ele in rel_info["drug_id"]],
                                 dtype=np.int64),
                        np.array([ele for ele in rel_info["disease_id"]],
                                 dtype=np.int64))
        rating_values = rel_info["values"].values.astype(np.float32)

        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False,
                            train_drug_indices=None, train_disease_indices=None,
                            drug_graph=None, disease_graph=None):
        data_dict = dict()

        num_train_drug = len(train_drug_indices) if train_drug_indices is not None else self._num_drug
        num_train_disease = len(train_disease_indices) if train_disease_indices is not None else self._num_disease

        if train_drug_indices is not None:
            global_drug_id_to_train = {gid: lid for lid, gid in enumerate(train_drug_indices)}
            rating_row = np.array([global_drug_id_to_train[gid] for gid in rating_pairs[0]])
        else:
            rating_row = rating_pairs[0]

        if train_disease_indices is not None:
            global_disease_id_to_train = {gid: lid for lid, gid in enumerate(train_disease_indices)}
            rating_col = np.array([global_disease_id_to_train[gid] for gid in rating_pairs[1]])
        else:
            rating_col = rating_pairs[1]

        num_nodes_dict = {'drug': num_train_drug, 'disease': num_train_disease}
        for rating in self.possible_rel_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating_name = to_etype_name(rating)
            data_dict.update({
                ('drug', str(rating_name), 'disease'): (rrow, rcol),
                ('disease', 'rev-%s' % str(rating_name), 'drug'): (rcol, rrow)
            })

        if add_support:
            data_dict.update({
                ('drug', 'drug_sim', 'drug'): ([], []),  # 定义药物相似性边
                ('disease', 'disease_sim', 'disease'): ([], [])  # 定义疾病相似性边
            })

        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        if add_support and drug_graph is not None and disease_graph is not None:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)

            drug_src, drug_dst = drug_graph.coalesce().indices()
            graph.add_edges(drug_src, drug_dst, etype=('drug', 'drug_sim', 'drug'))

            disease_src, disease_dst = disease_graph.coalesce().indices()
            graph.add_edges(disease_src + num_train_drug, disease_dst + num_train_drug,
                            etype=('disease', 'disease_sim', 'disease'))

            drug_ci = graph.in_degrees(etype=('drug', 'drug_sim', 'drug'))
            disease_ci = graph.in_degrees(etype=('disease', 'disease_sim', 'disease'))
            drug_ci = _calc_norm(drug_ci)
            disease_ci = _calc_norm(disease_ci)

            graph.nodes['drug'].data.update({'ci': drug_ci})
            graph.nodes['disease'].data.update({'ci': disease_ci})

        return graph

    def _generate_dec_graph(self, rating_pairs, train_drug_indices=None, train_disease_indices=None):
        num_train_drug = len(train_drug_indices) if train_drug_indices is not None else self.num_drug
        num_train_disease = len(train_disease_indices) if train_disease_indices is not None else self.num_disease

        if train_drug_indices is not None:
            global_drug_id_to_train = {gid: lid for lid, gid in enumerate(train_drug_indices)}
            mapped_drug_ids = np.array([global_drug_id_to_train[gid] for gid in rating_pairs[0]])
        else:
            mapped_drug_ids = rating_pairs[0]

        if train_disease_indices is not None:
            global_disease_id_to_train = {gid: lid for lid, gid in enumerate(train_disease_indices)}
            mapped_disease_ids = np.array([global_disease_id_to_train[gid] for gid in rating_pairs[1]])
        else:
            mapped_disease_ids = rating_pairs[1]

        ones = np.ones_like(mapped_drug_ids)
        drug_disease_rel_coo = sp.coo_matrix(
            (ones, (mapped_drug_ids, mapped_disease_ids)),
            shape=(num_train_drug, num_train_disease), dtype=np.float32
        )

        g = dgl.bipartite_from_scipy(drug_disease_rel_coo, utype='drug', etype='rate', vtype='disease')

        return g

    @property
    def num_links(self):
        return self.possible_rel_values.size

    @property
    def num_disease(self):
        return self._num_disease

    @property
    def num_drug(self):
        return self._num_drug