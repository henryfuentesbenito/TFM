import numpy as np
import torch
from torch.utils.data import Dataset
import argparse, os

class SmallSynthData(Dataset):
    def __init__(self, data_path, mode, params):
        self.mode = mode
        self.data_path = data_path
        if self.mode == 'train':
            path = os.path.join(data_path, 'train_feats')
            edge_path = os.path.join(data_path, 'train_edges')
        elif self.mode == 'val':
            path = os.path.join(data_path, 'val_feats')
            edge_path = os.path.join(data_path, 'val_edges')
        elif self.mode == 'test':
            path = os.path.join(data_path, 'test_feats')
            edge_path = os.path.join(data_path, 'test_edges')
        self.feats = torch.load(path)
        self.edges = torch.load(edge_path)
        self.same_norm = params['same_data_norm']
        self.no_norm = params['no_data_norm']
        if not self.no_norm:
            self._normalize_data()

    def _normalize_data(self):
        train_data = torch.load(os.path.join(self.data_path, 'train_feats'))
        if self.same_norm:
            self.feat_max = train_data.max()
            self.feat_min = train_data.min()
            self.feats = (self.feats - self.feat_min)*2/(self.feat_max-self.feat_min) - 1
        else:
            self.loc_max = train_data[:, :, :, :2].max()
            self.loc_min = train_data[:, :, :, :2].min()
            self.feats[:,:,:, :2] = (self.feats[:,:,:,:2]-self.loc_min)*2/(self.loc_max - self.loc_min) - 1

    def unnormalize(self, data):
        if self.no_norm:
            return data
        elif self.same_norm:
            return (data + 1) * (self.feat_max - self.feat_min) / 2. + self.feat_min
        else:
            result1 = (data[:, :, :, :2] + 1) * (self.loc_max - self.loc_min) / 2. + self.loc_min
            return result1

    def __getitem__(self, idx):
        return {'inputs': self.feats[idx], 'edges':self.edges[idx]}

    def __len__(self):
        return len(self.feats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_train', type=int, default=100)
    parser.add_argument('--num_val', type=int, default=100)
    parser.add_argument('--num_test', type=int, default100)
    parser.add_argument('--num_time_steps', type=int, default=50)
    parser.add_argument('--pull_factor', type=float, default=0.1)
    parser.add_argument('--push_factor', type=float, default=0.05)

    args = parser.parse_args()
    np.random.seed(1)

    # Supongamos que tus datos ya están almacenados en archivos
    train_path = os.path.join(args.output_dir, 'train_feats')
    val_path = os.path.join(args.output_dir, 'val_feats')
    test_path = os.path.join(args.output_dir, 'test_feats')
    train_edges_path = os.path.join(args.output_dir, 'train_edges')
    val_edges_path = os.path.join(args.output_dir, 'val_edges')
    test_edges_path = os.path.join(args.output_dir, 'test_edges')

    # Cargar los datos
    train_data = torch.load(train_path)
    val_data = torch.load(val_path)
    test_data = torch.load(test_path)
    train_edges = torch.load(train_edges_path)
    val_edges = torch.load(val_edges_path)
    test_edges = torch.load(test_edges_path)

    # Guardar los datos cargados en las rutas especificadas
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    torch.save(test_data, test_path)
    torch.save(train_edges, train_edges_path)
    torch.save(val_edges, val_edges_path)
    torch.save(test_edges, test_edges_path)
