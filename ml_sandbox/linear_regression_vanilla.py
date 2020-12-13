import torch

a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)


def f(x):
    return a * x + b


x = torch.rand(1000)
for i in range(1000):
    loss = (f(x) - (6 * x - 4)).square().mean()
    loss.backward()

    a.data -= a.grad.data * 0.1
    b.data -= b.grad.data * 0.1
    a.grad.zero_()
    b.grad.zero_()

    print(a.item(), b.item(), loss.item())
