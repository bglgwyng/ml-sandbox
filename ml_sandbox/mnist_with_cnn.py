import gzip
import math
import pickle
import shutil
from pathlib import Path

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = Path("data")
IMAGE_PATH = Path("image")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"


device = torch.device("cuda")

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    if not IMAGE_PATH.exists():
        IMAGE_PATH.mkdir(parents=True, exist_ok=True)
        for j, i in enumerate(x_valid):
            pyplot.imsave(IMAGE_PATH / f"{j}.png", i.reshape((28, 28)), cmap="gray")
    x_train, y_train, x_valid, y_valid = map(
        lambda x: torch.tensor(x).to(device), (x_train, y_train, x_valid, y_valid)
    )


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        xb = xb.view(-1, xb.size(1))
        return xb


def preprocess(x):
    return x.view(-1, 1, 28, 28)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            loss_sum = 0
            count_sum = 0
            for xb, yb in valid_dl:
                loss, count = loss_batch(model, loss_func, xb, yb)
                loss_sum += loss * count
                count_sum += count
        valid_loss = loss_sum / count_sum

        print(epoch, valid_loss)


train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)

m = 28 * 28

lr = 0.5

epochs = 5
bs = 64


loss_func = F.cross_entropy


def get_model():
    model = Mnist_Logistic()
    model.to(device)
    return model, optim.SGD(model.parameters(), lr=lr)


def get_data(train_ds, valid_ds, bs):
    return (DataLoader(train_ds, bs, shuffle=True), DataLoader(valid_ds, bs * 2))


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

print(accuracy(model(x_valid), y_valid))
