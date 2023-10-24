import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from load_data import load_mol_eval_single_dataset, load_mol_eval_multi_dataset
from model import SingleMLP, MultiMLP

seed = 1
epochs = 100
log_interval = 10
batch_size_train = 1000
batch_size_test = 1000

mol_eval_train_path = '../data/MoleculeEvaluationData/train.pkl.gz'
mol_eval_test_path = '../data/MoleculeEvaluationData/test.pkl.gz'

# device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
single_mlp = SingleMLP().to(device)
multi_mlp = MultiMLP().to(device)
optimizer = torch.optim.Adam(single_mlp.parameters(), lr=1e-3)
loss_func = torch.nn.MSELoss()

# print(next(single_mlp.parameters()).device)

single_train_dataset = load_mol_eval_single_dataset(mol_eval_train_path)
single_test_dataset = load_mol_eval_single_dataset(mol_eval_test_path)
single_train_loader = DataLoader(dataset=single_train_dataset, batch_size=batch_size_train, shuffle=True)
single_test_loader = DataLoader(dataset=single_test_dataset, batch_size=batch_size_test, shuffle=True)

multi_train_dataset = load_mol_eval_multi_dataset(single_train_dataset)
multi_test_dataset = load_mol_eval_multi_dataset(single_test_dataset)
multi_train_loader = DataLoader(dataset=multi_train_dataset, batch_size=batch_size_train, shuffle=True)
multi_test_loader = DataLoader(dataset=multi_test_dataset, batch_size=batch_size_test, shuffle=True)

train_costs = []
for i in range(len(single_train_dataset.dataset)):
    train_costs.append(single_train_dataset.dataset[i][1].item())
test_costs = []
for i in range(len(single_test_dataset.dataset)):
    test_costs.append(single_test_dataset.dataset[i][1].item())

# plt.hist(train_costs, bins=50, density=True, color="r", alpha=1)
# plt.hist(test_costs, bins=50, density=True, color="b", alpha=0.5)
# plt.xlabel("cost")
# plt.ylabel("frequency")
# plt.legend(["train", "test"])
# plt.xlim(0, 15)
# plt.title("The cost distribution of different datasets", fontdict={'color': 'k'})
# # plt.show()
# plt.savefig("cost_distribution.pdf")
