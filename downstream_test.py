import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils.metrics import Accuracy
from utils.scheduler import WarmupCosineLR
from utils.regularizer import L2EmbedPenalty
from model import TabularBERT, MLP
from sklearn.model_selection import train_test_split
import pandas as pd
import rtdl
from utils.data import QuantileDiscretize
import tqdm


device = torch.device('cuda:0')
data = pd.read_csv("/home/beom/GMT/datasets/GE.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = pd.Categorical(y).codes.astype(int)

train_X, test_X, train_labels, test_labels = train_test_split(X, y, train_size = 0.8, random_state = 0)
train_X, valid_X, train_labels, valid_labels = train_test_split(train_X, train_labels, train_size = 0.8, random_state = 0)

num_bins = 50

discretizer = QuantileDiscretize(num_bins = num_bins)
discretizer.fit(train_X)

train_bin_ids = discretizer.discretize(train_X)
valid_bin_ids = discretizer.discretize(valid_X)
test_bin_ids = discretizer.discretize(test_X)

class DownstreamData(Dataset):
    def __init__(self, bin_data, y, device):
        super(DownstreamData, self).__init__()
        self.bin_data = bin_data
        self.y = y
        self.device = device
        
    def __getitem__(self, idx):
        bin_idx = self.bin_data[idx]
        bin_idx = torch.tensor(bin_idx, device = self.device)
        label = torch.tensor(self.y[idx], device = self.device, dtype = torch.long)
        return bin_idx, label
    
    def __len__(self):
        return len(self.y)


train = DownstreamData(train_bin_ids, train_labels, device)
valid = DownstreamData(valid_bin_ids, valid_labels, device)
test = DownstreamData(test_bin_ids, test_labels, device)
trainloader = DataLoader(train, 
                         batch_size = 256,
                         shuffle = True)
validloader = DataLoader(valid, 
                         batch_size = len(valid), 
                         shuffle = True,
                         drop_last = False)
testloader = DataLoader(test,
                        batch_size = len(test),
                        shuffle = False,
                        drop_last = False)

class DownstreamMLP(nn.Module):
    def __init__(self, pretrained, output_dim):
        super(DownstreamMLP, self).__init__()
        self.embedding = pretrained.embedding
        self.bert = pretrained.bert
        # self.pooler = BertPooler(self.bert.embedding_dim)
        self.output_dim = output_dim
        # self.mlp = nn.Linear(32, 1)
        # self.mlp = nn.ModuleList()
        # for j in range(self.model.embedding_dim):
        #     self.mlp.append(nn.Linear(32, 1))
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p = 0.1)
        # self.fc = nn.Sequential(
        #     # nn.LayerNorm(self.bert.d_model),
        #     nn.Linear(self.model.d_model, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p = 0.5),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Dropout(p = 0.5),
        #     nn.Linear(32, output_dim)
        # )
        # self.hidden = Hidden(self.model.embedding_dim * 32)
        # self.fc = MLP(self.hidden, output_dim)
        # self.fc = rtdl.MLP(d_in = self.embedding.embedding_dim * 32,
        #                    d_out = output_dim, 
        #                    d_layers = [self.embedding.embedding_dim, self.embedding.embedding_dim], 
        #                 #    d_layers = [512, 512], 
        #                    activation = nn.ReLU, dropouts = 0.1)
        self.fc = MLP(input_dim = self.embedding.embedding_dim,
                      output_dim = output_dim, 
                      hidden_layers = [self.embedding.embedding_dim], 
                      activation = nn.ReLU, dropouts = 0.3)
        
        
        # self.fc = nn.Linear(self.embedding.embedding_dim,
        #                     output_dim)
        # self.fc.weight.data.normal_(mean = 0.0, std = 0.010)
        # self.fc.bias.data.zero_()
        
        
    def forward(self, bin_ids):
        out = self.embedding(bin_ids)
        out = self.bert(out)
        # out = self.pooler(out[:, 0])
        # out = self.fc(out)
        # out = self.fc(out[:, 0])
        # out = self.relu(out[:, 0])
        # out = self.dropout(out)
        # out_list = list()
        # for j in range(out.size(-1)):
        #     out_list.append(self.mlp[j](out[:, :, j]))
        # out = torch.cat(out_list, dim = -1)
        # out = self.mlp(out[:, 1:].transpose(1, 2)).squeeze(-1)
        # out = self.relu(out)
        # out = self.dropout(out)
        # out = self.fc(out.mean(dim = 1))
        out = self.fc(out[:, 0])
        # out = self.fc(out[:, 1:].flatten(1))
        # out = self.fc(out)
        # out = out.flatten(1)
        # out = self.fc(out)
        return out
    
pretrained_model = torch.load("/home/beom/GMT/pretraining/version1/model_checkpoint.pt")
model = DownstreamMLP(pretrained_model, 5)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 1e-4,
                              weight_decay = 1e-5,
                            #   momentum = 0.9,
                            #   nesterov = True
                            )

criterion = nn.CrossEntropyLoss()
penalty = L2EmbedPenalty(lamb = 0.5)
accuracy = Accuracy()
EPOCHS = 1000
total_steps = EPOCHS * len(trainloader)
scheduler = WarmupCosineLR(optimizer, 
                           warmup_epochs = 1,
                           max_epochs = total_steps, 
                           eta_min = 1e-6,
                           warmup_start_lr = 1e-5)

for epoch in tqdm.tqdm(range(EPOCHS)):
    risk = 0.
    acc_cum = 0.
    for i, (train_bin_ids, train_labels) in enumerate(trainloader):
        model.train(); optimizer.zero_grad()
        preds = model(train_bin_ids)
        
        # loss = criterion(preds, train_labels) 
        loss = criterion(preds, train_labels) + penalty(model.embedding.bin_embedding.weight)
        # loss = criterion(preds, train_labels) + penalty(model.embedding.bin_embeddings)
        # loss = loss1
        
        acc = accuracy(preds, train_labels)
        # acc = torch.sum(preds.argmax(dim = -1) == train_labels) / len(train_labels)
        acc_cum += acc
        
        loss.backward(); optimizer.step(); scheduler.step()
        risk += loss.item()
        # logger.add_scalar('Loss/train', loss.item(), k)
        # logger.add_scalar('Accuracy/train', acc, k)
        
        
    print('EPOCH: {epoch}, Training Loss: {loss}, Accuracy: {acc}'.format(epoch = epoch, 
                                                                          loss = risk / (i + 1), 
                                                                          acc = acc_cum / (i + 1)))
    # print(loss1)
    # print('EPOCH: {epoch}, Training Loss: {loss}, Level acc: {lev_acc}, Sublevel acc: {sublev_acc}'.format(epoch = epoch, loss = risk / (i + 1),
    #                                                                                                        lev_acc = lev_acc / (i + 1), sublev_acc = sublev_acc / (i + 1)))
    with torch.no_grad():
        model.eval()
        for j, (valid_bin_ids, valid_labels) in enumerate(validloader):
            valid_preds = model(valid_bin_ids)
            # val_loss1 = criterion1(valid_pred_level.transpose(1, 2), valid_y[valid_level_ids == 1].view(1974, -1))
            # valid_sublevel_y = valid_y[valid_level_ids == 2].view(1974, -1)
            # valid_sublevel_y[valid_sublevel_y != -100] = valid_sublevel_y[valid_sublevel_y != -100] - 100
            # val_loss2 = criterion2(valid_pred_sublevel.squeeze(2).squeeze(2).transpose(1, 2), valid_sublevel_y)
            # val_loss = val_loss1 + val_loss2
            val_loss = criterion(valid_preds, valid_labels)
            # val_loss = val_loss2
            
            # pred_val_class = valid_pred_y.argmax(axis = 1)
            # valid_acc += torch.sum(pred_val_class == valid_y)
            valid_risk = val_loss.item()
            val_acc = accuracy(valid_preds, valid_labels)
            # val_acc = torch.sum(valid_preds.argmax(dim = -1) == valid_labels) / len(valid_labels)
            
            # logger.add_scalar('Loss/valid', val_loss.item(), epoch)
            # logger.add_scalar('Accuracy/valid', val_acc, epoch)
            # checkpoint(val_acc, model, epoch)
        print('Validation Loss: {valid_loss}, Validation Accuracy: {valid_acc}'.format(valid_loss = valid_risk,
                                                                                     valid_acc = val_acc))    




with torch.no_grad():
    model.eval()
    for j, (test_bin_ids, test_labels) in enumerate(testloader):
        test_preds = model(test_bin_ids)
        # val_loss1 = criterion1(valid_pred_level.transpose(1, 2), valid_y[valid_level_ids == 1].view(1974, -1))
        # valid_sublevel_y = valid_y[valid_level_ids == 2].view(1974, -1)
        # valid_sublevel_y[valid_sublevel_y != -100] = valid_sublevel_y[valid_sublevel_y != -100] - 100
        # val_loss2 = criterion2(valid_pred_sublevel.squeeze(2).squeeze(2).transpose(1, 2), valid_sublevel_y)
        # val_loss = val_loss1 + val_loss2
        test_loss = criterion(test_preds, test_labels)
        # val_loss = val_loss2
        
        # pred_val_class = valid_pred_y.argmax(axis = 1)
        # valid_acc += torch.sum(pred_val_class == valid_y)
        test_risk = test_loss.item()
        test_acc = accuracy(test_preds, test_labels)
        # val_acc = torch.sum(valid_preds.argmax(dim = -1) == valid_labels) / len(valid_labels)
        
        # logger.add_scalar('Loss/valid', val_loss.item(), epoch)
        # logger.add_scalar('Accuracy/valid', val_acc, epoch)
        # checkpoint(val_acc, model, epoch)
    print('Test Loss: {test_loss}, test Accuracy: {test_acc}'.format(test_loss = test_risk,
                                                                                    test_acc = test_acc)) 