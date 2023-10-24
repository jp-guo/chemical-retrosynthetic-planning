import torch
from global_var import device, loss_func
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch import nn


def test(model: nn.Module, test_loader: DataLoader) -> float:
    size = len(test_loader)
    model.eval()
    test_loss = 0
    y_true = []
    y_pred = []
    test_r2 = 0
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.float().to(device), target.to(device)
            output = model(input)
            test_loss += loss_func(output, target).item()
            # Collect true and predicted values for R-squared calculation
            y_true.extend(target.tolist())
            y_pred.extend(output.tolist())
            # Calculate R-squared for the test set
            test_r2 += r2_score(y_true, y_pred)
            # print(f"Test R^2: {test_r2:.4f}")
    test_loss /= size
    test_r2 /= size
    print('\nTest set: Avg. loss: {:.4f}\n'.format(test_loss))
    print(f"Test R^2: {test_r2:.4f}")
    return test_loss


def test_multi_1(model: nn.Module, test_loader: DataLoader) -> float:
    size = len(test_loader)
    model.eval()
    test_loss = 0
    y_true = []
    y_pred = []
    test_r2 = 0
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.float().to(device), target.to(device)
            output = model(input)
            # output = output.squeeze()
            indices = RandomSampler(range(len(output)), replacement=False, num_samples=3)
            indices = [i for i in indices]
            output_sum = 0
            target_sum = 0
            for i in indices:
                output_sum += output[i]
                target_sum += target[i]
            test_loss += loss_func(output_sum, target_sum).item()
            # Collect true and predicted values for R-squared calculation
            y_true.extend(target.tolist())
            y_pred.extend(output.tolist())
            # Calculate R-squared for the test set
            test_r2 += r2_score(y_true, y_pred)
            # print(f"Test R^2: {test_r2:.4f}")
    test_loss /= size
    test_r2 /= size
    print('\nTest set: Avg. loss: {:.4f}\n'.format(test_loss))
    print(f"Test R^2: {test_r2:.4f}")
    return test_loss
