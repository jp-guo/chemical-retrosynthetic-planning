import torch
from torch import nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


class RolloutPolicyNet(nn.Module):
    def __init__(self, n_rules, fp_dim=2048, dim=64,
                 dropout_rate=0.5):
        super(RolloutPolicyNet, self).__init__()
        self.fp_dim = fp_dim
        self.n_rules = n_rules
        self.dropout_rate = dropout_rate
        self.fc1 = nn.Linear(fp_dim,dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(dim,n_rules)

    def forward(self, x):
        x = self.dropout1(F.elu(self.bn1(self.fc1(x))))
        x = self.fc2(x)

        return x


class MLP(nn.Module):
    def __init__(self, n_rules=13144, model_dump=None):
        super(MLP, self).__init__()
        self.model = RolloutPolicyNet(n_rules)
        if model_dump is not None:
            print(self.model.load_state_dict(torch.load(model_dump)))

    def morgan_fingerprint(self, product):
        '''
            Product is a chemical molecule, which can be transformed into a Morgan FingerPrint vector by the library rdkit as follows.
        '''
        mol = Chem.MolFromSmiles(product)

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

        onbits = list(fp.GetOnBits())
        arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
        arr[onbits] = 1

        return arr

    def forward(self, raw_prod):
        x = self.morgan_fingerprint(raw_prod).reshape(1, -1)
        x = self.model(torch.from_numpy(x).float().cuda())
        return x


# raw_product = "[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:15]1[c:14]2[cH:13][cH:12][c:11]([C:9]([CH3:8])=[O:10])[cH:19][c:18]2[cH:17][cH:16]1"
# mlp = MLP(model_dump='logs/single_step-dim-512-lr0.0001-bsz256/it40.pth').cuda()
# mlp.eval()
# print(mlp(raw_product))



