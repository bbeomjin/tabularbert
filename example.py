import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from model import TabularBERTTrainer

data = pd.read_csv("./datasets/GE.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = pd.Categorical(y).codes.astype(int)

train_X, valid_X, train_labels, valid_labels = train_test_split(X, y, test_size = 0.2, random_state = 0)

scaler = QuantileTransformer(n_quantiles = 10000,
                             output_distribution = 'uniform',
                             subsample = None)
# scaler = StandardScaler()
scaler.fit(train_X)
train_XX = scaler.transform(train_X)
valid_XX = scaler.transform(valid_X)


model = TabularBERTTrainer(x = train_XX, 
                           num_bins = 50,
                           encoding_info = None, 
                           device = torch.device('cuda:0'), 
                           valid_x = valid_XX)
model.setup_directories_and_logging(phase = 'pretraining', use_wandb = False)
model.set_bert(embedding_dim = 1024,
              n_layers = 3,
              n_heads = 8)
model.pretrain(lamb = 0.5,
               mask_token_prob = 1,
               random_token_prob = 0.2,
               unchanged_token_prob = 0.79,
               num_workers = 4)