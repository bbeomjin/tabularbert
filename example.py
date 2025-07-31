import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from model.TabularBERT import TabularBERTTrainer

data = pd.read_csv("./datasets/GE.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = pd.Categorical(y).codes.astype(int)

train_X, valid_X, train_labels, valid_labels = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = TabularBERTTrainer(x = train_X, 
                           num_bins = 50,
                           encoding_info = None, 
                           device = torch.device('cuda:0'), 
                           valid_x = valid_X)
model.setup_directories_and_logging(use_wandb = False)
model.set_mlm(embedding_dim = 1024,
              n_layers = 3,
              n_heads = 8)
model.pretrain(lamb = 0.5)