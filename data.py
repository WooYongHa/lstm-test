import torch
import glob
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)

#Todo(YongHa) make this as GPU Calculation
def findFiles(path): return glob.glob(path)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = filename.split('\\')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

def letterToIndex(letter):
    return all_letters.find(letter)

def lineToTensor(lines):
    batch_size = len(lines)
    max_len = max(list(len(str) for str in lines))
    tensor = torch.zeros(max_len, batch_size, n_letters)
    for line_idx, line in enumerate(lines):
        for letter_idx, letter in enumerate(line):
            tensor[letter_idx][line_idx][letterToIndex(letter)] = 1
    tensor = tensor.view(max_len, batch_size, -1)
    return tensor