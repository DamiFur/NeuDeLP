from process_input import get_train_test_datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import config
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import argparse


class Net(nn.Module):

    def __init__(self, num_layers=3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear((config.ARGUMENT_SIZE * 2) + 1, 120)
        self.layers = []
        for i in range(num_layers):
            self.layers.append(nn.Linear(120, 120))
        self.fc7 = nn.Linear(120, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = F.sigmoid(self.fc7(x))
        return x


def train_epoch(model, opt, criter, train_X, train_Y):
    # TODO: Implement batch_size
    losses = []
    model.train()

    for input, target in zip(train_X, train_Y):
        opt.zero_grad()
        input = torch.FloatTensor(input)
        target = torch.FloatTensor([target])
        predicted = model(input)
        loss = criter(predicted, target)
        loss.backward()
        opt.step()
        l = loss.data.numpy()
        losses.append(l)

    return losses


def test_model(model, test_data):
    model.eval()
    preds = []
    truth = []
    for input, target in test_data:
        input = torch.FloatTensor(input)
        target = torch.FloatTensor([target])
        predicted = model(input)

        preds.append(round(float(predicted[0])))
        truth.append(round(float(target)))

    return [accuracy_score(truth, preds), precision_score(truth, preds), recall_score(truth, preds), f1_score(truth, preds)]


parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="Learning rate value", type=float, default=2e-03)
parser.add_argument("--num_layers", help="number of internal layers for the neural net", type=int, default=5)
parser.add_argument("--epochs", help="Amount of epochs for the training process", type=int, default=10)
parser.add_argument("--complexity", help="Complexity of the programs generated for training the NN", type=str, default="simple", choices=["simple", "medium", "complex"])
args = parser.parse_args()

net = Net(num_layers=args.num_layers)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr)

acum_losses = []
epochs = args.epochs

X_train, X_test, Y_train, Y_test = get_train_test_datasets(args.complexity)

for e in range(epochs):
    acum_losses += train_epoch(net, optimizer, criterion, X_train, Y_train)

print(test_model(net, list(zip(X_test, Y_test))))

# plt.plot(acum_losses)
# plt.savefig("test.png")
# plt.show()