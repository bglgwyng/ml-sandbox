import torch
import torch.optim as optim

a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

optimizer = optim.SGD([a, b], lr=0.1)


def f(x):
    return a * x + b


x = torch.rand(1000)
for i in range(1000):
    loss = (f(x) - (6 * x - 4)).square().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(a.item(), b.item(), loss.item())
