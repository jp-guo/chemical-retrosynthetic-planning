import time
import pickle
import gzip


def get_dataset():
    starting_mols_root = r'../data/Multi-Step task/starting_mols.pkl.gz'
    target_mol_route_root = r'../data/Multi-Step task/target_mol_route.pkl'
    test_mols_root = r'../data/Multi-Step task/test_mols.pkl'

    s1 = time.time()
    # starting_mols = pickle.load(gzip.open(starting_mols_root, "rb") ,encoding='latin1')
    # starting_mols = set(starting_mols)
    e1 = time.time()

    s2 = time.time()
    with open(target_mol_route_root,'rb') as f:
        target_mol_route = pickle.load(f)
    e2 = time.time()

    s3 = time.time()
    with open(test_mols_root, 'rb') as f:
        target_mols = pickle.load(f)
    e3 = time.time()
    
    # print(f'Time of loading:starting mol:{e1-s1:.2f} target mol route:{e2-s2:.2f} target mol:{e3-s3:.2f}')
    # # print(f'Starting mol:{len(starting_mols)}')
    # print(f'Target mols:{len(target_mols)}')
    print(target_mols[0])

    # return starting_mols, target_mol_route, target_mols
    return target_mols
C[C@H](c1ccccc1)N1C[C@]2(C(=O)OC(C)(C)C)C=CC[C@@H]2C1=S
if __name__ == '__main__':
    get_dataset()
    # s = time.time()
    # starting_mols, target_mol_route, target_mols = get_dataset()
    # e = time.time()
    
    # print(f'Time:{e-s:.2f}')
    
    
    # # performance test
    # s = time.time()
    # for _ in range(100):
    #     cnt = 0
    #     for mol in target_mols:
    #         if mol in starting_mols:
    #             cnt += 1
    # e = time.time()
    
    # print(f'Time:{e-s:.2f}')
    # print(cnt)
    
    