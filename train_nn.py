from process_input import get_train_test_datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import config
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import argparse
import statistics


class Net(nn.Module):

    def __init__(self, num_layers=3, input_size=(config.ARGUMENT_SIZE * 2) + 1, output_size = 2, layers_size=300):
        super(Net, self).__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, layers_size)
        self.layers = []
        for i in range(num_layers):
            self.layers.append(nn.Linear(layers_size, layers_size))
        if output_size == 2:
            self.output_layer = nn.Linear(layers_size, 1)
        else:
            self.output_layer = nn.Linear(layers_size, 3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        if self.output_size == 2:
            x = F.sigmoid(self.output_layer(x))
        else:
            x = F.softmax(self.output_layer(x), dim=0)
            # print(x)
            # x = torch.argmax(x, dim=0)
            # print(x)
        return x

def train_epoch(model, opt, criter, train_X, train_Y):
    # TODO: Implement batch_size
    losses = []
    model.train()

    for input, target in zip(train_X, train_Y):
        opt.zero_grad()
        input = torch.FloatTensor(input)
        if model.output_size != 2:
            if target == 0:
                target = torch.FloatTensor([1., 0., 0.])
            elif target == 1:
                target = torch.FloatTensor([0., 1., 0.])
            else:
                target = torch.FloatTensor([0., 0., 1.])
        else:
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

        #print("predicted: {} - target: {}".format(predicted, target))
        prd = -1
        trth = -1
        if model.output_size != 2:
            prd = float(torch.argmax(predicted, dim=0))
            trth = float(target[0])

        else:
            prd = round(float(predicted[0]))
            trth = round(float(target[0]))

        preds.append(prd)
        truth.append(trth)

    if model.output_size != 2:
        return [accuracy_score(truth, preds), precision_score(truth, preds, average='macro'), recall_score(truth, preds, average='macro'), f1_score(truth, preds, average='macro')]
    return [accuracy_score(truth, preds), precision_score(truth, preds), recall_score(truth, preds), f1_score(truth, preds)]
    # with blocking 
    # return [accuracy_score(truth, preds), precision_score(truth, preds, average='micro'), recall_score(truth, preds, average='micro'), f1_score(truth, preds, average='micro')]


parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="Learning rate value", type=float, default=2e-03)
parser.add_argument("--num_layers", help="number of internal layers for the neural net", type=int, default=5)
parser.add_argument("--epochs", help="Amount of epochs for the training process", type=int, default=20)
parser.add_argument("--complexity", help="Complexity of the programs generated for training the NN", type=str, default="simple", choices=["simple", "medium", "complex"])
parser.add_argument("--blocking", help="Does program have blocking arguments?", type=bool, default=False)
parser.add_argument("--presumptions", help="Does program have presumptions?", type=bool, default=False)
parser.add_argument("--program_size", help="Size of program taken as input", type=int, default=1000)
parser.add_argument("--output_size", help="Two or three classes classification?", type=int, default=2)
parser.add_argument("--arg_size", help="Size of the argument. Use -1 for using the longest argument length", type=int, default=-1)
parser.add_argument("--layers_size", help="Size of inner layers", type=int, default=300)
args = parser.parse_args()
print(args)

presumptions = "presumption_enabled" if args.presumptions else "presumption_disabled"
datasets, max_arg_size = get_train_test_datasets(complexity=args.complexity, blocking=args.blocking, program_size=args.program_size, output_size=args.output_size, max_arg_size=args.arg_size, presumptions=presumptions)
net = Net(num_layers=args.num_layers, input_size=2*max_arg_size+1, output_size=args.output_size, layers_size=args.layers_size)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr)

acum_losses = []
epochs = args.epochs


accuracy = []
precision = []
recall = []
f1 = []
w = open("results/res_{},{},{},{},{},{},{},{}".format(args.lr, args.num_layers, args.complexity, args.blocking, args.output_size, presumptions, args.layers_size, args.program_size), "w")
for X_train, X_test, Y_train, Y_test in datasets:
    for e in range(epochs):
        acum_losses += train_epoch(net, optimizer, criterion, X_train, Y_train)

    net.eval()
    metrics_aux = test_model(net, list(zip(X_test, Y_test)))
    accuracy.append(metrics_aux[0])
    precision.append(metrics_aux[1])
    recall.append(metrics_aux[2])
    f1.append(metrics_aux[3])

l = len(datasets)
print("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(args.lr, args.num_layers, args.complexity, args.blocking, args.output_size, presumptions, args.layers_size, args.program_size, statistics.mean(accuracy), statistics.mean(precision), statistics.mean(recall), statistics.mean(f1), statistics.stdev(f1)))
w.write("{}, {}, {}, {}, {}".format(statistics.mean(accuracy), statistics.mean(precision), statistics.mean(recall), statistics.mean(f1), statistics.stdev(f1)))


# plt.plot(acum_losses)
# plt.savefig("test.png")
# plt.show()
