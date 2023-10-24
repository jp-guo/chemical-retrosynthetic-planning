from global_var import device, optimizer, loss_func, log_interval
from torch.utils.data import DataLoader
from torch import nn


def train(model: nn.Module, train_loader: DataLoader, epoch: int):
    size = len(train_loader.dataset)
    model.train()
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target = input.float().to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(input),
                size,
                100. * batch_idx / len(train_loader),
                loss.item()
            ))
