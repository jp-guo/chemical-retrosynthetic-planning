from global_var import epochs, single_train_loader, single_test_loader, multi_train_loader, multi_test_loader, \
    single_mlp, multi_mlp
from train import train
from test import test, test_multi_1


def single_eval():
    for epoch in range(epochs):
        train(single_mlp, single_train_loader, epoch)
    test(single_mlp, single_test_loader)


def multi_eval_1():
    for epoch in range(epochs):
        train(single_mlp, single_train_loader, epoch)
    test_multi_1(single_mlp, single_test_loader)


def multi_eval_2():
    for epoch in range(epochs):
        train(multi_mlp, multi_train_loader, epoch)
    test(multi_mlp, multi_test_loader)


if __name__ == '__main__':
    single_eval()
    multi_eval_1()
    multi_eval_2()
