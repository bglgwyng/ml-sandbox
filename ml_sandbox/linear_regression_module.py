import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))

    def forward(self, x):
        return self.a * x + self.b


net = Net()
net.to(device)
optimizer = optim.SGD(net.parameters(), lr=0.1)

x = torch.rand(1000, device=device)
for i in range(1000):
    loss = (net(x) - (6 * x - 4)).square().mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for j, k in net.named_parameters():
        print(j, k.item())
    print("loss", loss.item())
    print()
