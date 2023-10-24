import os.path
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from rdchiral.template_extractor import extract_from_reaction
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdchiral.main import rdchiralRunText
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pickle


def extract_template(reaction):
    reactants, products=reaction.split('>>')

    inputRec={
        '_id':None,
        'reactants':reactants,
        'products':products
        }

    ans = extract_from_reaction(inputRec)

    if 'reaction_smarts' in ans.keys():
        return ans['reaction_smarts']
    else:
        return None


def morgan_fingerprint(product):
    '''
        Product is a chemical molecule, which can be transformed into a Morgan FingerPrint vector by the library rdkit as follows.
    '''
    mol = Chem.MolFromSmiles(product)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048)

    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool_)
    arr[onbits]=1

    return arr


def recover(template, product):
    '''
    If you want to recover the reaction from the template, following code is helpful. Sometimes, the reaction can not be recovered, just skip this reaction.
    SMARTS string
    '''
    out=rdchiralRunText(template, product)
    return out


def build():
    path = 'schneider50k'
    template_dict = {}
    for split in ['test', 'train', 'val']:
        df = pd.read_csv(os.path.join(path, f'raw_{split}.csv'))
        X, y = torch.zeros(len(df), 2048), torch.zeros(len(df))
        retro_templates = list(df['reactants>reagents>production'])
        print(f'Building {split} set...')
        for i, reaction in enumerate(tqdm(retro_templates)):
            template = extract_template(reaction)
            if template is None: continue
            _, product = reaction.split('>>')
            try: recover(template, product)
            except Exception: continue
            X[i] = torch.from_numpy(morgan_fingerprint(product)).float()
            if template in template_dict.keys():
                y[i] = template_dict[template]
            else:
                template_dict[template] = len(template_dict)
                y[i] = template_dict[template]

        save_dir = os.path.join('fingerprint', split)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(X, os.path.join(save_dir, 'features.pth'))
        torch.save(y, os.path.join(save_dir, 'labels.pth'))
    print(f'Total templates: {len(template_dict)}')


class SinglestepDataset():
    def __init__(self, path=None):
        super(SinglestepDataset, self).__init__()
        if path is not None:
            self.X = torch.load(os.path.join(path, 'features.pth'))
            self.y = torch.load(os.path.join(path, 'labels.pth'))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def merge(self, *sets):
        self.X = sets[0].X
        self.y = sets[0].y
        n = len(sets)
        for i in range(1, n):
            self.X = torch.cat((self.X, sets[i].X), dim=0)
            self.y = torch.cat((self.y, sets[i].y), dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')

    args = parser.parse_args()

    if args.build:
        build()