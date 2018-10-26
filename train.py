import torch
from data import *
from model import *
import random
import time
import math

#TODO(YongHa) Make this value as Configure file
n_hidden = 200
n_epochs = 500000
batch_size = 32
print_every = 200
plot_every = 1000
learning_rate = 0.005
num_layer = 4

#For GPU Operation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(batch_size):
    categories, lines = [], []
    for i in range(batch_size):
        category = randomChoice(all_categories)
        line = randomChoice(category_lines[category])
        categories.append(all_categories.index(category))
        lines.append(line)
    category_tensor = torch.tensor(categories, dtype=torch.long)
    line_tensor = lineToTensor(lines)
    return categories, lines, category_tensor.to(device=device), line_tensor.to(device=device)

#TODO(YongHa) Make this parameter as cofigure file
lstm = LSTM(n_letters, n_hidden, n_categories, batch_size, num_layer)
optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def train(category_tensor, line_tensor):
    hidden, cell = lstm.init_hidden()
    optimizer.zero_grad()
    output, hidden, cell = lstm(line_tensor, hidden, cell)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    return output, loss.item()

current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

if __name__ == '__main__':
    for iter in range(1, n_epochs + 1):
        categories, lines, category_tensor, line_tensor = randomTrainingExample(batch_size)
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # print number, loss, name, prediction
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            print('%d %d%% (%s) %.4f %s' % (iter, iter / n_epochs * 100, timeSince(start), loss, lines))

        # add currnet loss to loss list
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

torch.save(lstm, 'char-lstm-classification.pt')