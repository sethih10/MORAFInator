import numpy as np
import h5py

import torch
from torch.utils.data import Dataset

COVALENT_RADII = {
    1: 0.32,
    6: 0.77,
    7: 0.75,
    8: 0.73,
    9: 0.71,
}

ELEMENT_TO_INDEX = {
    1: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
}

INDEX_TO_SYMBOL = {
    0: 'H',
    1: 'C',
    2: 'N',
    3: 'O',
    4: 'F',
}



class AFMData(Dataset): 
    def __init__(self, data_path, tokenizer, transform, train_size=0.8, split='train'): 
        self.data_path = data_path
        self.transform = transform
        self.split = split
        self.tokenizer = tokenizer

        with h5py.File(self.data_path, 'r') as f:
            total_length = f['x'].shape[0]
            self.train_length = int(train_size*total_length)
            self.val_length = total_length - self.train_length

    def __len__(self): 
        if self.split == 'train': 
            return self.train_length
        else: 
            return self.val_length

    def __getitem__(self, idx):

        if self.split == 'train': 
            idx += 0
        else: 
            idx += self.train_length

        with h5py.File(self.data_path, 'r') as f:
            x = f['x'][idx]
            xyz = f['xyz'][idx]

        # Remove padding atoms
        xyz = xyz[xyz[:, -1] > 0]

        # Get edges 
        edges = []
        for i in range(xyz.shape[0]): 
            for j in range(i+1, xyz.shape[0]): 
                dist = np.linalg.norm(xyz[i, :3] - xyz[j, :3])
                if dist < 1.2*(COVALENT_RADII[xyz[i, -1]] + COVALENT_RADII[xyz[j, -1]]):
                    edges.append([i,j, -100])

        # Normalize xyz to [0.25, 0.75]
        xyzmin = np.min(xyz[:, :3])
        xyz_max = np.max(xyz[:, 3])

        xyz[:, :3] = (xyz[:, :3] - xyzmin)/(xyz_max - xyzmin)
        xyz[:, :3] = 0.5*xyz[:, :3] + 0.25

        # map atom types to integers (0,1, ..)
        xyz[:, -1] = [ELEMENT_TO_INDEX[atom_type] for atom_type in xyz[:, -1]]

        #sample = {'coords': xyz, edges: np.asarray(edges)}

        #if self.transform:
        #    sample = self.transform(sample)

        # Keep all channels from HDF5 and convert to [C,H,W]
        x = torch.from_numpy(x)
        if x.dim() == 4 and x.size(0) == 1: 
            # Input like [1, H, W, C] -> [H, W, C]
            x = x.squeeze(0)
        if x.dim() == 3: 
            # Assume [H, W, C] from HDF5, move channels first
            x = x.permute(2,0,1).contiguous()

        mask = xyz[:, -1] > 0
        atomtok = " ".join([INDEX_TO_SYMBOL[value] for value in xyz[mask, -1]])

        #atomtok = self.tokenizer['atomtok'].text_to_sequence(atomtok)

        coord_bin_values = np.round(np.linspace(0,1,64),2)

        nodes = {'coords': [], 'symbols': [], 'smiles': "", 'indices': []}
        for i, (symbol, coord) in enumerate(zip(atomtok, xyz)):
            new_coord = coord_bin_values[np.argmin(np.abs(coord_bin_values.reshape(1,-1) - coord.reshape(-1,1)), axis = 1)]
            #atomtok_coords.append(f'{symbol}: {new_coord[0]}, {new_coord[1]}, {new_coord[2]},')
            nodes['coords'].append(list(new_coord[:2]))
            nodes['smiles'] += symbol + " "
            nodes['symbols'].append(symbol)
            nodes['indices'].append(i+1)

        atomtok_coords = self.tokenizer['atomtok_coords'].nodes_to_sequence(nodes)

        

        #ref = {'atomtok': atomtok, 'edges': np.asarray(edges), 'atomtok_coords': atomtok_coords, 'chartok_coords': atomtok_coords}
        #ref = {'atomtok_coords': atomtok_coords, 'atom_indices': , 'coords': nodes['coords'], 'edges': edges}
        ref = {'atomtok_coords': atomtok_coords, 'edges':edges, 'coords': nodes['coords'], 'atom_indices': nodes['indices']}
        return idx, x, ref


def get_datasets(data_path, tokenizer, train_transform = None, val_transform = None, train_size=0.8):

    train_dataset = AFMData(data_path, tokenizer = tokenizer, transform=train_transform, train_size=train_size, split='train')
    val_dataset = AFMData(data_path, tokenizer = tokenizer, transform=val_transform, train_size=train_size, split='val')

    return train_dataset, val_dataset


def afm_collate_fn(batch):

    #sample = {'coords':[], 'edges':[]}
    ref = {'atomtok': [], 'edges': [], 'atomtok_coords': [], 'chartok_coords': []}
    ref = {'atomtok_coords': [], 'edges': [], 'coords': [], 'atom_indices': []}
    #ref = {'atomtok_coords': []}

    PAD_ID = 0
    length = 128
    MAX_ATOMS = 60
    pad_coord = [[-1, -1]]
    pad_edges = [[-100, -100, -100]]
    

    ids = [id[0] for id in batch]
    images = torch.stack([item[1] for item in batch])

    toks = []
    coordss = []
    atom_indicess = []
    edgess = []
    for item in batch:
        #sample['coords'].append(torch.from_numpy(item[2]['coords']))
        #sample['edges'].append(torch.from_numpy(item[2]['edges']))
        #ref['atomtok'].append(torch.from_numpy(item[2]['atomtok']))
        #ref['edges'].append(torch.from_numpy(item[2]['edges']))

        tok = item[2]['atomtok_coords']
        
        pad = [PAD_ID]*(length - len(tok))
        tok.extend(pad)

        coords = item[2]['coords']

        pad = pad_coord*(MAX_ATOMS - len(coords))
        coords.extend(pad)

        atom_indices = item[2]['atom_indices']
        pad = [0]*(MAX_ATOMS - len(atom_indices))
        atom_indices.extend(pad)

        

        edges = item[2]['edges']
        n = MAX_ATOMS

        mat = torch.full((1, n, n), fill_value=-100, dtype=torch.long)
        symmetric = True
        for item in edges:
            u, v = int(item[0]), int(item[1])
            t = int(item[2]) if len(item) > 2 else default
            if 0 <= u < n and 0 <= v < n:
                mat[0, u, v] = t
                if symmetric:
                    mat[0, v, u] = t

        edges = mat
        #pad = pad_edges*(100 - len(edges))
        #edges.extend(pad)

        toks.append(tok)
        coordss.append(coords)
        atom_indicess.append(atom_indices)
        edgess.append(edges)
            
    ref['atomtok_coords'].append(torch.tensor(toks))
    ref['atomtok_coords'].append(torch.tensor([len(toks[0])]*len(toks)))

    ref['edges'] = torch.tensor(edges)
    
    ref['coords'] = torch.tensor(coordss)
    
    ref['atom_indices'].append(torch.tensor(atom_indicess))
    ref['atom_indices'].append(torch.tensor([len(atom_indicess[0])]*len(atom_indicess)))
    
        #ref['atomtok_coords'].append(torch.from_numpy(item[2]['atomtok_coords']))
        #ref['chartok_coords'].append(torch.from_numpy(item[2]['chartok_coords']))

    return ids, images, ref
    #return ids, images, sample