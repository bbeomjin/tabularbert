

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from model.GMTMLM import GMTModel

data = pd.read_csv("./datasets/GE.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = pd.Categorical(y).codes.astype(int)

train_X, valid_X, train_labels, valid_labels = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = GMTModel(x = train_X, 
                 encoding_info = {'K': 10, 'L': 10}, 
                 device = torch.device('cuda:0'), 
                 valid_x = valid_X)
model.set_mlm(embedding_dim = 512,
              n_heads = 8)
model.train(lamb = 1, criterion = 'was')