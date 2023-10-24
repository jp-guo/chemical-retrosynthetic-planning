import argparse
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.mlp import RolloutPolicyNet
from dropbox.dataset import SinglestepDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../dropbox/fingerprint')
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--batchSz', type=int, default=256)
    parser.add_argument('--testBatchSz', type=int, default=512)
    parser.add_argument('--nEpoch', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pretrained', type=str, default=None) # '/mnt/nas/home/guojinpei/Retrosynthetic-Planning/single_step/dropbox/mlp.pth'
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--save_dir', required=True)

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    save = args.save_dir

    os.makedirs(save, exist_ok=True)

    train_set = SinglestepDataset(os.path.join(args.data_dir, 'train'))
    val_set = SinglestepDataset(os.path.join(args.data_dir, 'val'))
    test_set = SinglestepDataset(os.path.join(args.data_dir, 'test'))

    train_val_set = SinglestepDataset()
    train_val_set.merge(train_set, val_set)

    print("number of all training data:", len(train_set))
    print("number of all validation data:", len(val_set))
    print("number of all testing data:", len(test_set))

    del train_set, val_set

    model = RolloutPolicyNet(13144, 2048, args.dim)

    if args.cuda: model = model.cuda()

    if args.pretrained is not None:
        print(model.load_state_dict(torch.load(args.pretrained)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    log_file = os.path.join(save, f'train_{cur_time}.txt') if not args.do_eval else os.path.join(save, f'eval_{cur_time}.txt')

    device = 'cuda' if args.cuda else 'cpu'
    if not args.do_eval:
        for epoch in range(1, args.nEpoch+1):
            train(epoch, model, optimizer, train_val_set, args.batchSz, log_file, device)
            test(epoch, model, optimizer, test_set, args.testBatchSz, log_file, device)
            torch.save(model.state_dict(), os.path.join(save, 'it'+str(epoch+1)+'.pth'))
    else:
        eval(model, test_set, args.testBatchSz, log_file, device)


def top_k_acc(preds, gt, k=1):
    probs, idx = torch.topk(preds, k=k)
    idx = idx.cpu().numpy().tolist()
    gt = gt.cpu().numpy().tolist()
    num = preds.size(0)
    correct = 0
    for i in range(num):
        for id in idx[i]:
            if id == gt[i]:
                correct += 1
                break
    return correct


def run(epoch, model, optimizer, dataset, batchSz, to_train=False, log_file=None, device='cuda'):
    loss_final = 0
    loader = DataLoader(dataset, batch_size=batchSz)
    tloader = tqdm(enumerate(loader), total=len(loader))
    k = 10
    acc_1_final = 0
    acc_k_final = 0
    for i, (X, y) in tloader:
        X, y = X.to(device), y.to(device)
        if to_train:
            model.train()
        else:
            model.eval()

        preds = model(X)
        loss = nn.functional.cross_entropy(preds, y.to(torch.int64))
        if to_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc_1 = top_k_acc(preds, y, 1)
        acc_k = top_k_acc(preds, y, k)
        acc_1_final += acc_1
        acc_k_final += acc_k
        tloader.set_description(
            'Epoch {} {} Loss {:.4f} Acc_1: {:.4f} Acc_{}: {:.4f}'.format(epoch, ('Train' if to_train else 'Test '), loss.item(), acc_1 / len(X), k,
                                                                                        acc_k / len(X)))
        loss_final += loss.item()

    loss_final = loss_final / len(loader)
    acc_1_final = acc_1_final / len(dataset)
    acc_k_final = acc_k_final / len(dataset)
    with open(log_file, 'a') as f:
        pre = 'Train' if to_train else 'Test'
        print(f'{pre} epoch:{epoch}, loss={loss_final}, Acc_1={acc_1_final:.4f}, Acc_{k}={acc_k_final:.4f}', file=f)
    if not to_train:
        print('TESTING SET RESULTS: Average loss: {:.4f} Acc_1: {:.4f} Acc_{}: {:.4f}'.format(loss.item(), acc_1_final, k, acc_k_final))
    else:
        print('TRAINING SET RESULTS: Average loss: {:.4f} Acc_1: {:.4f} Acc_{}: {:.4f}'.format(loss.item(), acc_1_final, k, acc_k_final))

    torch.cuda.empty_cache()


def train(epoch, model, optimizer, dataset, batchSz, log_file, device):
    run(epoch, model, optimizer, dataset, batchSz, True, log_file, device)


@torch.no_grad()
def test(epoch, model, optimizer, dataset, batchSz, log_file, device):
    run(epoch, model, optimizer, dataset, batchSz, False, log_file, device)


@torch.no_grad()
def eval(model, dataset, batchSz, log_file, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=batchSz)
    tloader = tqdm(enumerate(loader), total=len(loader))
    ks = [1, 3, 5, 10, 20, 50, 100]
    acc_ks_final = np.zeros(len(ks))
    for i, (X, y) in tloader:
        X, y = X.to(device), y.to(device)
        preds = model(X)

        for i, k in enumerate(ks):
            acc_ks_final[i] += top_k_acc(preds, y, k)

    acc_ks_final /= len(dataset)
    with open(log_file, 'a') as f:
        for i, k in enumerate(ks):
            print(f'top {k} acc={acc_ks_final[i]}', file=f)
            print(f'top {k} acc={acc_ks_final[i]}')
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()