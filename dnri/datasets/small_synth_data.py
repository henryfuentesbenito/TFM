import numpy as np
import torch
from torch.utils.data import Dataset
from andi_datasets.models_phenom import models_phenom
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

def generate_and_store_data(T, N, L, D, alphas, Ds, r, Pb, Pu):
    all_data = []
    all_edges = []
    while len(all_data) < num_sims:
        trajs_model3, labels_model3 = models_phenom().dimerization(
            N=N,
            L=L,
            T=T,
            alphas=alphas,  # Exponentes anómalos
            Ds=Ds,  # Coeficientes de difusión
            r=r,  # Radio de las partículas
            Pb=Pb,  # Probabilidad de unión
            Pu=Pu  # Probabilidad de desunión
        )

        # Verificar dimerización en las dos primeras trayectorias
        has_dimerization_first_two = any(np.any(labels_model3[:, i, 0] == 0.9) for i in range(2))
        
        # Verificar que la tercera trayectoria no tenga dimerización
        no_dimerization_third = not np.any(labels_model3[:, 2, 0] == 0.9)
        
        if has_dimerization_first_two and no_dimerization_third:
            all_data.append(trajs_model3)
            all_edges.append(labels_model3[:, :, 0] == 0.9)
            
    return all_data, all_edges

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_train', type=int, default=100)
    parser.add_argument('--num_val', type=int, default=100)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--num_time_steps', type=int, default=50)
    parser.add_argument('--pull_factor', type=float, default=0.1)
    parser.add_argument('--push_factor', type=float, default=0.05)

    args = parser.parse_args()
    np.random.seed(1)
    num_sims = args.num_train + args.num_val + args.num_test
    
    T = args.num_time_steps  # Número de pasos de tiempo por trayectoria
    N = 3  # Número de trayectorias
    L = 1.5 * 12  # Tamaño de la caja (en píxeles)
    D = 0.1  # Coeficiente de difusión (en píxeles^2 / frame)
    alphas = [1.0, 0.9]  # Exponentes anómalos
    Ds = [10 * D, 1.0 * D]  # Coeficientes de difusión
    r = 1  # Radio de las partículas
    Pb = 1  # Probabilidad de unión
    Pu = 0.1  # Probabilidad de desunión (para que se separen después de unirse)

    all_data, all_edges = generate_and_store_data(T, N, L, D, alphas, Ds, r, Pb, Pu)

    all_data = np.stack(all_data)
    all_edges = np.stack(all_edges)

    train_data = torch.FloatTensor(all_data[:args.num_train])
    val_data = torch.FloatTensor(all_data[args.num_train:args.num_train+args.num_val])
    test_data = torch.FloatTensor(all_data[args.num_train+args.num_val:])
    train_path = os.path.join(args.output_dir, 'train_feats')
    torch.save(train_data, train_path)
    val_path = os.path.join(args.output_dir, 'val_feats')
    torch.save(val_data, val_path)
    test_path = os.path.join(args.output_dir, 'test_feats')
    torch.save(test_data, test_path)

    train_edges = torch.FloatTensor(all_edges[:args.num_train])
    val_edges = torch.FloatTensor(all_edges[args.num_train:args.num_train+args.num_val])
    test_edges = torch.FloatTensor(all_edges[args.num_train+args.num_val:])
    train_path = os.path.join(args.output_dir, 'train_edges')
    torch.save(train_edges, train_path)
    val_path = os.path.join(args.output_dir, 'val_edges')
    torch.save(val_edges, val_path)
    test_path = os.path.join(args.output_dir, 'test_edges')
    torch.save(test_edges, test_path)
