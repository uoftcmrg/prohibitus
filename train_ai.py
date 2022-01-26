import csv
from operator import itemgetter
from random import shuffle

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

dataset = []

with open('primes.csv') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        integer, primality = map(int, row.values())
        bin_digits = tuple((integer & (1 << i)) >> i for i in range(64))
        dataset.append((bin_digits, primality))

shuffle(dataset)

n = len(dataset)
X = torch.Tensor(tuple(map(itemgetter(0), dataset)))
y = F.one_hot(torch.LongTensor(tuple(map(itemgetter(1), dataset)))) \
    .to(torch.float32)
train_test_split = 0.8

split_index = int(n * train_test_split)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]
input_dim = len(X[0])
output_dim = len(y[0])

epoch_count = 10000
learning_rate = 0.005

model = torch.nn.Linear(input_dim, output_dim)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

report_epoch_count = 100
epochs = []
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []


def stats_of(inputs, labels):
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        accuracy = (outputs.argmax(1) == labels.argmax(1)).sum() \
                   / labels.shape[0]

    return loss.item(), accuracy


def report_stats(epoch):
    epochs.append(epoch)

    train_loss, train_accuracy = stats_of(X_train, y_train)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    test_loss, test_accuracy = stats_of(X_test, y_test)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(
        f'[Epoch {epoch + 1:04d}]',
        f'Train loss: {train_loss},',
        f'Train accuracy: {train_accuracy * 100:.2f}%,',
        f'Test loss: {test_loss},',
        f'Test accuracy: {test_accuracy * 100:.2f}%,',
    )


for epoch in range(epoch_count):
    if epoch % report_epoch_count == 0:
        report_stats(epoch)

    inputs, labels = X_train, y_train

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
else:
    report_stats(epoch_count)

plt.title('Train and Test Loss vs Epoch')
plt.plot(epochs, train_losses, label='Train')
plt.plot(epochs, test_losses, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.title('Train and Test Accuracies vs Epoch')
plt.plot(epochs, train_accuracies, label='Train')
plt.plot(epochs, test_accuracies, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

with torch.no_grad():
    print(model(torch.Tensor([1, 0, 0, 1] + [0] * 60)).argmax())
    print(model(torch.Tensor([0, 0, 0, 1] + [0] * 60)).argmax())
    print(model(torch.Tensor([0, 0, 0, 0] + [1] * 60)).argmax())
    print(model(torch.Tensor([1, 0, 0, 0] + [1] * 60)).argmax())
