from process_input import get_input_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self, num_layers=3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(81, 120)
        self.layers = []
        for i in range(num_layers):
            self.layers.append(nn.Linear(120, 120))
        self.fc7 = nn.Linear(120, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = F.sigmoid(self.fc7(x))
        # x = self.fc4(x)
        return x


def train_epoch(model, opt, criter, batch_size=50):
    # TODO: Implement batch_size
    losses = []
    model.train()
    for input, target in get_input_data():
        # print(input)
        # print(target)
        opt.zero_grad()
        input = torch.FloatTensor(input)
        target = torch.FloatTensor([target])
        print("input")
        print(input)
        print("output")
        print(target)
        predicted = model(input)
        print("PRED")
        print(predicted)
        loss = criter(predicted, target)
        loss.backward()
        opt.step()
        print("==========================================")
        l = loss.data.numpy()
        print(l)
        losses.append(l)

    return losses


net = Net(num_layers=5)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=2e-03)

acum_losses = []
epochs = 10
for e in range(epochs):
    acum_losses += train_epoch(net, optimizer, criterion)

plt.plot(acum_losses)
plt.savefig("test.png")
plt.show()