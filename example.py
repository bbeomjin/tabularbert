import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from tabularbert import TabularBERTTrainer
from tabularbert.utils.metrics import Accuracy

# Load and preprocess data
data = pd.read_csv("./datasets/GE.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = pd.Categorical(y).codes.astype(int)

train_X, test_X, train_labels, test_labels = train_test_split(X, y, train_size = 0.8, random_state = 0)

# Pretraining
trainer = TabularBERTTrainer(x=train_X,
                             num_bins=50,
                             encoding_info=None,
                             device=torch.device('cuda:0'))
trainer.setup_directories_and_logging(save_dir='./pretraining',
                                      phase='pretraining',
                                      project_name='GE data pretraining',
                                      use_wandb=False)
trainer.set_bert(embedding_dim=1024,
                 n_layers=3,
                 n_heads=8)
trainer.pretrain(lamb=0.5,
                 mask_token_prob=1,
                 random_token_prob=0.2,
                 unchanged_token_prob=0.79,
                 num_workers=0)

# Finetuning
train_X, valid_X, train_labels, valid_labels = train_test_split(train_X, train_labels, train_size=0.8, random_state=0)
# If a pretrained model is available, load it
# trainer = TabularBERTTrainer.from_pretrained(save_path = './pretraining/version0/model_checkpoint.pt',
#                                              device = torch.device('cuda:0'))
trainer.setup_directories_and_logging(save_dir='./fine-tuning',
                                      phase='fine-tuning',
                                      project_name='GE data fine-tuning',
                                      use_wandb=False)
trainer.finetune(x=train_X,
                 y=train_labels,
                 valid_x=valid_X,
                 valid_y=valid_labels,
                 epochs=1000,
                 batch_size=256,
                 criterion=nn.CrossEntropyLoss(),
                 metric=Accuracy(ignore_index=-100),
                 num_workers=0)


