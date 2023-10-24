from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import os
import argparse
import pickle
import time
import numpy as np

from dropbox.dataset import SinglestepDataset

def top_k_accuracy(y_pred, y_true, k):
    top_k_preds = y_pred.argsort()[:, -k:]
    matches = 0
    for i in range(len(y_true)):
        if y_true[i].int().item() in top_k_preds[i]:
            matches += 1
    return matches / len(y_true)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../dropbox/fingerprint')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--kernel', default='linear')
    parser.add_argument('--C', type=float, default=0.0001)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--do_eval', action='store_true')

    args = parser.parse_args()

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

    if not args.do_eval:
        model = OneVsRestClassifier(SVC(kernel=args.kernel, C=args.C, probability=True))
        model.fit(train_val_set.X[:10000], train_val_set.y[:10000].float())
        s = pickle.dumps(model)
        f = open(f'{save}/svm.model', "wb+")
        f.write(s)
        f.close()
        preds = model.predict(test_set.X)
        accuracy = accuracy_score(preds, test_set.y)

        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        file = os.path.join(save, f'train_{cur_time}.txt')

        with open(file, 'w') as f:
            print(f'kernel={args.kernel}, C={args.C}', file=f)
            print(f'accuracy={accuracy}', file=f)
        print(f'accuracy={accuracy}')
    else:
        f = open(args.pretrained, 'rb')
        s = f.read()
        model = pickle.loads(s)
        preds = model.predict_proba(test_set.X)
        ks = [1, 3, 5, 10, 20, 50, 100]
        acc_list = []
        for k in ks:
            acc = top_k_accuracy(preds, test_set.y, k)
            acc_list.append(acc)
            print(f'top-{k} acc={acc}')

        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        file = os.path.join(save, f'train_{cur_time}.txt')
        with open(file, 'w') as f:
            for i, acc in enumerate(acc_list):
                print(f'top-{ks[i]}={acc}', file=f)

main()